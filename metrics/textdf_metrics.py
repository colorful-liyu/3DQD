#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh
import trimesh.sample
# from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch
import os
import mcubes
import einops
from einops import rearrange, repeat
from skimage import measure
from termcolor import cprint
from tqdm.auto import tqdm
from tqdm import trange
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
# from StructuralLosses.match_cost import match_cost
from metrics.pvd_metrics.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from metrics.pvd_metrics.PyTorchEMD.emd import earth_mover_distance as EMD
from metrics.evaluation_fpd.FPD import calculate_fpd
from metrics.uncond_metrics import convert_points, emd_approx, _pairwise_EMD_CD_, _one_EMD_CD_
import clip
from PIL import Image

# used to render img for clip
from utils.util_3d import init_mesh_renderer
import pytorch3d
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.transforms import RotateAxisAngle
import imageio
import torchvision.utils as vutils


cham3D = chamfer_3DDist()


'''
用法
results = compute_all_metrics(device, gen_sdf, gt_sdf, batch_size)    计算mmd, cov, 1nn
'''

def compute_text_metrics(device, gen_sdf, gt_sdf, text_list, batch_size, mode, logger, EMD_flag, num_points=2048, save_flag=False, save_dir='text_samples/'):
    torch.cuda.set_device(device)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    dist, elev, azim = 1.7, 20, 20
    mesh_renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=device)

    sample_pcs, ref_pcs = convert_points(device, gen_sdf, gt_sdf, num_points, mode, logger)
    sample_pcs = sample_pcs.reshape(-1, batch_size, sample_pcs.shape[-2], sample_pcs.shape[-1])

    gen_sdf = gen_sdf.reshape(-1, batch_size, 1, 64, 64, 64)

    sample_pcs_min = []
    results = {'PMMD':0, 'ave_list_CD':0, 'min_list_EMD':0, 'ave_list_EMD':0, 'TMD':0, 'clip-s-cd-min':0, 'clip-s':0}
    print("Pairwise EMD CD")

    for i in trange(len(ref_pcs)):
        min_cd_val, min_cd_idx, ave_cd_val, min_emd_val, min_emd_idx, ave_emd_val = _one_EMD_CD_(sample_pcs[i], ref_pcs[i].unsqueeze(0), batch_size, EMD_flag, False)
        results['PMMD'] += min_cd_val
        results['ave_list_CD'] += ave_cd_val
        results['min_list_EMD'] += min_emd_val
        results['ave_list_EMD'] += ave_emd_val
        sample_pcs_min.append(sample_pcs[i, min_cd_idx, :, :].unsqueeze(0))
        all_cd, all_emd = _pairwise_EMD_CD_(sample_pcs[i], sample_pcs[i], batch_size, False, False)
        results['TMD'] += torch.mean(all_cd)

        _, mesh_for_render = sdf_to_mesh_for_metrics_two_output(gen_sdf[i, :, :, :, :, :])
        # mesh_to_img(device, mesh_renderer, mesh_for_render, text_list[i], note='genall')
        # _, mesh_for_render_gt = sdf_to_mesh_for_metrics(gt_sdf[i, :, :, :, :].unsqueeze(0))
        # mesh_to_img(device, mesh_renderer, mesh_for_render_gt, text_list[i], note='gt')
        # _, mesh_for_render = sdf_to_mesh_for_metrics(gen_sdf[i, min_cd_idx, :, :, :, :].unsqueeze(0))
        views_list = mesh_to_img(device, mesh_renderer, mesh_for_render, text_list[i], note='gen', save_flag=save_flag, save_dir=save_dir)  # bs/1, 20, [256, 256, 3]
        

        text_inputs = clip.tokenize(['A chair with'+text_list[i]]).to(device)
        text_features = clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        all_img_list=[]

        for h in range(len(views_list)):
            img_list=[]
            for j in range(len(views_list[h])):
                image_input = preprocess(Image.fromarray(views_list[h][j])).unsqueeze(0).to(device)
                img_list.append(image_input)
            img_list = torch.cat(img_list, dim=0)
            all_img_list.append(img_list)
            
            if h == min_cd_idx:
                image_features = clip_model.encode_image(img_list)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T)
                values, indices = similarity[:,0].topk(1, dim=0)
                results['clip-s-cd-min'] += (values[0])
        all_img_list = torch.cat(all_img_list, dim=0)
        image_features = clip_model.encode_image(all_img_list)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)
        values, indices = similarity[:,0].topk(1, dim=0)
        results['clip-s'] += (values[0])
        
    results = {k:(v.item()/len(ref_pcs)) for k, v in results.items()}

    sample_pcs_min = torch.cat(sample_pcs_min, dim=0)
    print(sample_pcs_min.shape, ref_pcs.shape)

    fpd = calculate_fpd(sample_pcs_min, ref_pcs, batch_size=len(sample_pcs_min), device=device)
    results['FPD'] = fpd

    print(results)

    return results



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


def sdf_to_mesh_for_metrics_two_output(sdf, level=0.04, color=None, render_all=False):

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

    p3d_mesh_list2 = pytorch3d.structures.Meshes(verts, faces, textures=pytorch3d.renderer.Textures(verts_rgb=verts_rgb)).to(device)

    return p3d_mesh_list, p3d_mesh_list2



def mesh_to_img(device, mesh_renderer, gen_mesh, text, note='', save_flag=False, save_dir='text_samples/'):
    """ setup renderer """
    gen_gif_name = os.path.join(save_dir, f'{text}_{note}.gif'.replace('/', ''))
    views_list = save_mesh_as_gif(mesh_renderer, gen_mesh, nrow=3, out_name=gen_gif_name, device=device, save_flag=save_flag)
    return views_list

def save_mesh_as_gif(mesh_renderer, mesh, save_flag, nrow=3, out_name='1.gif', device='cuda'):
    """ save batch of mesh into gif """
    # img_comb = render_mesh(mesh_renderer, mesh, norm=False)    
    # rotate
    rot_comb = rotate_mesh_20(mesh_renderer, mesh, device) # save the first one
    
    if save_flag:
        # gather img into batches
        nimgs = len(rot_comb)
        nrots = len(rot_comb[0])
        H, W, C = rot_comb[0][0].shape
        rot_comb_img = []
        for i in range(nrots):
            img_grid_i = torch.zeros(nimgs, H, W, C)
            for j in range(nimgs):
                img_grid_i[j] = torch.from_numpy(rot_comb[j][i])
                
            img_grid_i = img_grid_i.permute(0, 3, 1, 2)
            img_grid_i = vutils.make_grid(img_grid_i, nrow=nrow)
            img_grid_i = img_grid_i.permute(1, 2, 0).numpy().astype(np.uint8)
                
            rot_comb_img.append(img_grid_i)

        with imageio.get_writer(out_name, mode='I', duration=.08) as writer:
            
            # combine them according to nrow
            for rot in rot_comb_img:
                writer.append_data(rot)

    return rot_comb  # list of 20 views image

def rotate_mesh_20(mesh_renderer, mesh, device):
    cur_mesh = mesh

    B = len(mesh.verts_list())
    ret = [ [] for i in range(B)]

    for i in range(20):
        cur_mesh = rotate_mesh(cur_mesh, device='cpu')
        img = render_mesh(mesh_renderer, cur_mesh, norm=False, device=device) # b c h w # important!! no norm here or they will not align
        img = img.permute(0, 2, 3, 1) # b h w c
        img = img.detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        for j in range(B):
            ret[j].append(img[j, :, :, 0:3])

    return ret

def rotate_mesh(mesh, axis='Y', angle=18, device='cuda'):
    rot_func = RotateAxisAngle(angle, axis, device=device)

    verts = mesh.verts_list()
    faces = mesh.faces_list()
    textures = mesh.textures
    
    B = len(verts)

    rot_verts = []
    for i in range(B):
        v = rot_func.transform_points(verts[i])
        rot_verts.append(v)
    new_mesh = Meshes(verts=rot_verts, faces=faces, textures=textures)
    return new_mesh

def render_mesh(renderer, mesh, color=None, norm=True, device='cuda'):
    # verts: tensor of shape: B, V, 3
    # return: image tensor with shape: B, C, H, W
    if mesh.textures is None:
        verts = mesh.verts_list()
        verts_rgb_list = []
        for i in range(len(verts)):
        # print(verts.min(), verts.max())
            verts_rgb_i = torch.ones_like(verts[i])
            if color is not None:
                for i in range(3):
                    verts_rgb_i[:, i] = color[i]
            verts_rgb_list.append(verts_rgb_i)

        texture = pytorch3d.renderer.Textures(verts_rgb=verts_rgb_list)
        mesh.textures = texture
    images = renderer(mesh.to(device))
    return images.permute(0, 3, 1, 2)