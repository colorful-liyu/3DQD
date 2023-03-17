#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh
import trimesh.sample
# from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch
import h5py
import mcubes
import einops
from einops import rearrange, repeat
from skimage import measure
from termcolor import cprint
from tqdm import trange
from metrics.uncond_metrics import convert_points, _pairwise_EMD_CD_, emd_approx
from metrics.pvd_metrics.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from metrics.pvd_metrics.PyTorchEMD.emd import earth_mover_distance as EMD_method

cham3D = chamfer_3DDist()

def directed_hausdorff(point_cloud1:torch.Tensor, point_cloud2:torch.Tensor, reduce_mean=True):
    """
    :param point_cloud1: (B, 3, N)
    :param point_cloud2: (B, 3, M)
    :return: directed hausdorff distance, A -> B
    """
    n_pts1 = point_cloud1.shape[2]
    n_pts2 = point_cloud2.shape[2]

    pc1 = point_cloud1.unsqueeze(3)
    pc1 = pc1.repeat((1, 1, 1, n_pts2)) # (B, 3, N, M)
    pc2 = point_cloud2.unsqueeze(2)
    pc2 = pc2.repeat((1, 1, n_pts1, 1)) # (B, 3, N, M)
    
    l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1)) # (B, N, M)
    shortest_dist, _ = torch.min(l2_dist, dim=2)

    hausdorff_dist, _ = torch.max(shortest_dist, dim=1) # (B, )

    if reduce_mean:
        hausdorff_dist = torch.mean(hausdorff_dist)

    return hausdorff_dist


def compute_all_shape_comp_metrics(gt_sdf, comp_sdf, comp_type, mode):
    results = {'ave_CD':0, 'ave_EMD':0, 'TMD':0, 'MMD_CD':0, 'MMD_EMD':0, 'UHD':0}

    for i in trange(len(gt_sdf)):
        aveCD, aveEMD, TMCD, MMD_CD, MMD_EMD_sameCD, UHD = evaluate_one_instance(gt_sdf[i], comp_sdf[i], comp_type, mode)
        results['ave_CD'] += aveCD
        results['ave_EMD'] += aveEMD
        results['TMD'] += TMCD
        results['MMD_CD'] += MMD_CD
        results['MMD_EMD'] += MMD_EMD_sameCD
        results['UHD'] += UHD

    results = {k:(v/len(gt_sdf)) for k, v in results.items()}

    print(results)

    return results


def evaluate_one_instance(gt_sdf, gen_sdf, comp_type='half', mode=None, partial_pcs=None):

    device = gen_sdf.device
    
    # calculated UHD with different comp_type
    if partial_pcs == None:
        with torch.no_grad():
            sample_pcs, ref_pcs = convert_points(gt_sdf.device, gen_sdf, gt_sdf, 8192, mode)
            sample_pcs = sample_pcs.cpu()
            ref_pcs = ref_pcs.cpu()

            if comp_type=='half':
                partial_pcs_mask = ref_pcs[:, :, 1]<=0
                partial_pcs = ref_pcs[partial_pcs_mask].unsqueeze(0).repeat(sample_pcs.shape[0],1,1)
                hausdorff = directed_hausdorff(partial_pcs.permute(0,2,1), sample_pcs.permute(0,2,1), reduce_mean=True).item()
            elif comp_type=='octant':
                partial_pcs_mask_x = ref_pcs[:, :, 0]<=0
                partial_pcs_mask_y = ref_pcs[:, :, 1]<=0
                partial_pcs_mask_z = ref_pcs[:, :, 2]<=0
                partial_pcs_mask = (partial_pcs_mask_x & partial_pcs_mask_y & partial_pcs_mask_z)
                partial_pcs = ref_pcs[partial_pcs_mask].unsqueeze(0).repeat(sample_pcs.shape[0],1,1)
                hausdorff = directed_hausdorff(partial_pcs.permute(0,2,1), sample_pcs.permute(0,2,1), reduce_mean=True).item()
            
    else:
        sample_pcs = gen_sdf.cpu()
        ref_pcs = gt_sdf.cpu()
        hausdorff = directed_hausdorff(partial_pcs.permute(0,2,1), sample_pcs.permute(0,2,1).to(gt_sdf.device), reduce_mean=True).item()

    ref_pcs = ref_pcs.to(device)
    sample_pcs = sample_pcs.to(device)


    bs = len(sample_pcs)
    dl, dr, _, _ = cham3D(sample_pcs, ref_pcs.expand(bs, -1, -1))
    chamfer_dist = (dl.mean(dim=1) + dr.mean(dim=1)).cpu().numpy()
    ref_exp = ref_pcs.expand(bs, -1, -1)

    earthmover_dist = emd_approx(sample_pcs, ref_exp).cpu().numpy()
    # If you want to get EMD, use the function below.
    # earthmover_dist = EMD_method(sample_pcs, ref_exp, transpose=False).cpu().numpy()

    CD = np.stack(chamfer_dist, axis=0)
    EMD = np.stack(earthmover_dist, axis=0)
    aveCD = np.mean(CD)
    aveEMD = np.mean(EMD)
    MMD_CD = np.min(CD)
    min_index = np.argmin(CD)
    MMD_EMD_sameCD = EMD[min_index]
    MMD_EMD= np.min(EMD)

    all_cd, all_emd = _pairwise_EMD_CD_(sample_pcs, sample_pcs, batch_size=16, EMD_flag = False, verbose = False)
    TMD = all_cd.mean()

    # # This is old computational method for TMD
    # TMD = 0 
    # for i in range(bs):

    #     mask = torch.ones(len(sample_pcs)).bool()
    #     mask[i] = False
    #     dl, dr, _, _ = cham3D(sample_pcs[mask], sample_pcs[i].unsqueeze(0).expand(bs-1, -1, -1))
    #     TMD += (dl.mean(dim=1) + dr.mean(dim=1)).mean().cpu().numpy()

    return aveCD, aveEMD, TMD, MMD_CD, MMD_EMD, hausdorff

