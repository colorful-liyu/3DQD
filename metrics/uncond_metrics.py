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
from tqdm.auto import tqdm
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
# from StructuralLosses.match_cost import match_cost
from metrics.pvd_metrics.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from metrics.pvd_metrics.PyTorchEMD.emd import earth_mover_distance as EMD
from lfd import LightFieldDistance
import os

cham3D = chamfer_3DDist()


'''
用法
results = compute_all_metrics(device, gen_sdf, gt_sdf, batch_size)    计算mmd, cov, 1nn
'''

def compute_all_metrics(device, gen_sdf, gt_sdf, batch_size, mode, logger, EMD_flag, num_points=2048, class_label=None):
    torch.cuda.set_device(device)
    sample_pcs, ref_pcs = convert_points(device, gen_sdf, gt_sdf, num_points, mode, logger, class_label)
    
    results = {}

    print("Pairwise EMD CD")
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, EMD_flag, verbose=True)

    ## CD
    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({
        "%s-CD" % k: v for k, v in res_cd.items()
    })
    
    if EMD_flag:
        # EMD
        res_emd = lgan_mmd_cov(M_rs_emd.t())
        results.update({
            "%s-EMD" % k: v for k, v in res_emd.items()
        })

    for k, v in results.items():
        print(k)
        print('[%s] %.8f' % (k, v.item()))

    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(ref_pcs, ref_pcs, batch_size, EMD_flag, verbose=True)
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(sample_pcs, sample_pcs, batch_size, EMD_flag, verbose=True)

    # 1-NN results
    ## CD
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update({
        "1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if 'acc' in k
    })
    if EMD_flag:
        # EMD
        one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
        results.update({
            "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
        })
    results = {k:v.item() for k, v in results.items()}

    jsd = jsd_between_point_cloud_sets(sample_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    results['jsd'] = jsd

    print(results)

    return results


def normalize_point_clouds(pcs, mode, class_label):

    if mode == None or mode == 'None':
        return pcs
    if mode == 'pf_norm':
        std_dict = {'car':[0.1635, 0.1544], 'chair':[0.1685, 0.1891], 'airplane':[0.1201, 0.1374]}
        pcs = pcs * std_dict[class_label][0] / std_dict[class_label][1]
        return pcs
    for i in range(pcs.size(0)):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        elif mode == 'l2norm':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = torch.max(torch.sqrt(torch.sum((pc-shift)**2, dim=1)))
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs


_EMD_NOT_IMPL_WARNED = False
def emd_approx(sample, ref):
    global _EMD_NOT_IMPL_WARNED
    emd = torch.zeros([sample.size(0)]).to(sample)
    if not _EMD_NOT_IMPL_WARNED:
        _EMD_NOT_IMPL_WARNED = True
        print('\n\n[WARNING]')
        print('  * EMD is not implemented due to GPU compatability issue.')
        print('  * We will set all EMD to zero by default.')
        print('  * You may implement your own EMD in the function `emd_approx` in ./evaluation/evaluation_metrics.py')
        print('\n')
    return emd

# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


def EMD_CD(sample_pcs, ref_pcs, batch_size, reduced=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    iterator = range(0, N_sample, batch_size)

    for b_start in tqdm(iterator, desc='EMD-CD'):
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        dl, dr, _, _ = cham3D(sample_batch, ref_batch)
        # dl, dr = distChamfer(sample_batch, ref_batch)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

        emd_batch = emd_approx(sample_batch, ref_batch)
        emd_lst.append(emd_batch)

    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)

    results = {
        'MMD-CD': cd,
        'MMD-EMD': emd,
    }
    return results

def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, EMD_flag, verbose=False):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(N_sample)
    if verbose:
        iterator = tqdm(iterator, desc='Pairwise EMD-CD')
    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        sub_iterator = range(0, N_ref, batch_size)
        # if verbose:
        #     sub_iterator = tqdm(sub_iterator, leave=False)
        for ref_b_start in sub_iterator:
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            point_dim = ref_batch.size(2)
            sample_batch_exp = sample_batch.view(1, -1, point_dim).expand(
                batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            dl, dr, _, _ = cham3D(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

            if EMD_flag:
                emd_batch = EMD(sample_batch_exp, ref_batch, transpose=False)
            else:
                emd_batch = emd_approx(sample_batch_exp, ref_batch)
            emd_lst.append(emd_batch.view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd


def _one_EMD_CD_(sample_pcs, ref_pcs, batch_size, EMD_flag, verbose=False):
    N_sample = sample_pcs.shape[0]

    cd_lst = []
    emd_lst = []
    iterator = range(0, N_sample, batch_size)

    if verbose:
        iterator = tqdm(iterator, desc='one EMD-CD')

    for sample_b_start in iterator:
        sample_b_end = min(N_sample, sample_b_start + batch_size)
        sample_batch = sample_pcs[sample_b_start:sample_b_end]

        batch_size_sample = sample_batch.size(0)
        point_dim = sample_batch.size(2)
        ref_batch_exp = ref_pcs.view(1, -1, point_dim).expand(
            batch_size_sample, -1, -1)
        ref_batch_exp = ref_batch_exp.contiguous()

        dl, dr, _, _ = cham3D(ref_batch_exp, sample_batch)
        cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(-1))

        if EMD_flag:
            emd_batch = EMD(ref_batch_exp, sample_batch, transpose=False)
        else:
            emd_batch = emd_approx(ref_batch_exp, sample_batch)
        emd_lst.append(emd_batch.view(-1))

    cd_lst = torch.cat(cd_lst, dim=0)
    emd_lst = torch.cat(emd_lst, dim=0)

    min_cd_val, min_cd_idx = torch.min(cd_lst, dim=0)
    ave_cd_val= torch.mean(cd_lst, dim=0)
    min_emd_val, min_emd_idx = torch.min(emd_lst, dim=0)
    ave_emd_val= torch.mean(emd_lst, dim=0)
    return min_cd_val, min_cd_idx, ave_cd_val, min_emd_val, min_emd_idx, ave_emd_val


# Adapted from https://github.com/xuqiantong/
# GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat([
        torch.cat((Mxx, Mxy), 1),
        torch.cat((Mxy.transpose(0, 1), Myy), 1)], 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(
        k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)

    # 对于生成的每个结果，找到最近的对应的reference点云，结果有N_sample个
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'num_ref_cov': torch.tensor(min_idx.unique().view(-1).size(0)),
        'lgan_mmd_smp': mmd_smp,
    }


def lgan_mmd_cov_match(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }, min_idx.view(-1)


#######################################################
# JSD : from https://github.com/optas/latent_3d_points
#######################################################
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with
    resolution^3 cells, that is placed in the unit-cube. If clip_sphere it True
    it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def jsd_between_point_cloud_sets(
        sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds,
       as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(
        sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(
        ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(
        pclouds, grid_resolution, in_sphere=False, verbose=False):
    """Given a collection of point-clouds, estimate the entropy of
    the random variables corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in tqdm(pclouds, desc='JSD'):
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    """another way of computing JSD"""

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))

    

def convert_points(device, gen_sdf, gt_sdf, num_points, mode, logger=None, class_label=None):

    
    if len(gen_sdf.shape) != 3:
        gen_mesh_list = sdf_to_mesh_for_metrics(gen_sdf)
        gen_points_sampled_list = []
        for i in range(len(gen_mesh_list)):
            gen_points_sampled = trimesh.sample.sample_surface(gen_mesh_list[i], num_points)[0]
            gen_points_sampled_list.append(torch.FloatTensor(gen_points_sampled).unsqueeze(0))
        gen_points_sampled_list = torch.cat(gen_points_sampled_list, dim=0)
        gen_points_sampled_list = normalize_point_clouds(gen_points_sampled_list, mode, class_label)
    else:
        gen_points_sampled_list = torch.FloatTensor(gen_sdf)
        
    gt_mesh_list = sdf_to_mesh_for_metrics(gt_sdf)
    gt_points_list = []
    for i in range(len(gt_mesh_list)):
        gt_points = trimesh.sample.sample_surface(gt_mesh_list[i], num_points)[0]
        gt_points_list.append(torch.FloatTensor(gt_points).unsqueeze(0))
    gt_points_list = torch.cat(gt_points_list, dim=0)
    gt_points_list = normalize_point_clouds(gt_points_list, mode, class_label)
    
    return gen_points_sampled_list.to(device), gt_points_list.to(device)


def sdf_to_mesh_for_metrics(sdf, level=0.04, color=None, render_all=False):
    # device='cuda'
    device=sdf.device

    # extract meshes from sdf
    n_cell = sdf.shape[-1]
    bs, nc = sdf.shape[:2]

    assert nc == 1

    nimg_to_render = bs
    # if not render_all:
    #     if bs > 16:
    #         cprint('Warning! Will not return all meshes', 'red')
    #     nimg_to_render = min(bs, 16) # no need to render that much..

    verts = []
    faces = []
    verts_rgb = []
    p3d_mesh_list = []

    for i in range(nimg_to_render):
        sdf_i = sdf[i, 0].detach().cpu().numpy()
        # verts_i, faces_i = mcubes.marching_cubes(sdf_i, 0.02)
        verts_i, faces_i = mcubes.marching_cubes(sdf_i, level)
        verts_i = verts_i / n_cell - .5 

        verts_i = torch.from_numpy(verts_i).float()
        faces_i = torch.from_numpy(faces_i.astype(np.int64))
        text_i = torch.ones_like(verts_i)
        if color is not None:
            for i in range(3):
                text_i[:, i] = color[i]

        verts.append(verts_i)
        faces.append(faces_i)
        verts_rgb.append(text_i)
        
        p3d_mesh = trimesh.Trimesh(vertices=verts_i, faces=faces_i, vertex_colors=text_i)
        p3d_mesh_list.append(p3d_mesh)


    return p3d_mesh_list

def lfd_smp_train(device, gen_sdf, gt_sdf, logger, save_dir):
    
    sample_pcs, ref_pcs = convert_points(device, gen_sdf, gt_sdf, 2048, 'None', logger)
    results = {}
    print("Pairwise EMD CD")
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(sample_pcs[:500] , ref_pcs, 48, False)
    print(M_rs_cd.shape)
    min_index = torch.argmin(M_rs_cd, dim=1)
    print(min_index)
    np.save(os.path.join(save_dir, 'CD_dist.npy'), M_rs_cd.cpu().numpy())
    np.save(os.path.join(save_dir, 'gen_sdf.npy'), gen_sdf.cpu().numpy())
    np.save(os.path.join(save_dir, 'gt_minCD_sdf.npy'), gt_sdf[min_index].cpu().numpy())
    print(gt_sdf[min_index].shape)
    # gen_mesh_list = sdf_to_mesh_for_metrics(gen_sdf)[:2] 
    # gt_mesh_list = sdf_to_mesh_for_metrics(gt_sdf)[:2] 
    # print(f'The num of gen mesh is {len(gen_mesh_list)}. The num of gt mesh is {len(gt_mesh_list)}')
    LFD_list = []
    n_cell = gen_sdf.shape[-1]

    for i in tqdm(range(len(min_index))):
        sdf_i = gen_sdf[i, 0].detach().cpu().numpy()
        # verts_i, faces_i = mcubes.marching_cubes(sdf_i, 0.02)
        verts_i, faces_i = mcubes.marching_cubes(sdf_i, 0.03)
        verts_i = verts_i / n_cell - .5 
        verts_i = torch.from_numpy(verts_i).float()
        faces_i = torch.from_numpy(faces_i.astype(np.int64))
        text_i = torch.ones_like(verts_i)
        # mesh --> .obj
        file_path = os.path.join(save_dir, f'gen_{i}.npy')
        mcubes.export_obj(verts_i, faces_i, file_path)
        gen_mesh = trimesh.Trimesh(vertices=verts_i, faces=faces_i, vertex_colors=text_i)
        
        sdf_i = gt_sdf[min_index[i], 0].detach().cpu().numpy()
        # verts_i, faces_i = mcubes.marching_cubes(sdf_i, 0.02)
        verts_i, faces_i = mcubes.marching_cubes(sdf_i, 0.03)
        verts_i = verts_i / n_cell - .5 
        verts_i = torch.from_numpy(verts_i).float()
        faces_i = torch.from_numpy(faces_i.astype(np.int64))
        text_i = torch.ones_like(verts_i)
        # mesh --> .obj
        file_path = os.path.join(save_dir, f'gt_{i}.npy')
        mcubes.export_obj(verts_i, faces_i, file_path)
        gt_mesh = trimesh.Trimesh(vertices=verts_i, faces=faces_i, vertex_colors=text_i)

        lfd_value: float = LightFieldDistance(verbose=True).get_distance(gen_mesh.vertices, gen_mesh.faces,gt_mesh.vertices, gt_mesh.faces)
        LFD_list.append(lfd_value)

    np.save(os.path.join(save_dir, 'lfd_dist.npy'), torch.tensor(LFD_list).numpy())
    print(LFD_list)
    

