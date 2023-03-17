
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

import torch

from datasets.base_dataset import CreateDataset
from datasets.dataloader import CreateDataLoader, get_data_generator

from models.base_model import create_model

import utils
from utils.qual_util import make_batch

class Opt:
    def __init__(self):
        self.name = 'opt'


def tensor_to_pil(tensor):
    """ assume shape: c h w """
    assert tensor.dim() == 3
    
    return Image.fromarray( (rearrange(tensor, 'c h w -> h w c').cpu().numpy() * 255.).astype(np.uint8) )

def get_shape_comp_opt(gpu_id=0, seed=333):
    opt = Opt()

    # args
    gpuid=[gpu_id]
    batch_size=1
    max_dataset_size=10000000

    name='test_diffusion'

    # default args
    opt.serial_batches = False
    opt.nThreads = 4

    # important args
    opt.dataset_mode = 'shapenet_code'
    opt.seed = seed
    opt.isTrain = False
    opt.gpu_ids = gpuid
    opt.device = 'cuda:%s' % gpuid[0]
    opt.batch_size = batch_size
    opt.max_dataset_size = max_dataset_size

    opt.name = name

    utils.util.seed_everything(opt.seed)
    opt.phase = 'test'
    return opt
    
def get_shape_comp_dset(opt):
    ##### setup dataset
    opt.ratio = 1.0

    # ShapeNet
    opt.cat = 'all'
    opt.trunc_thres = 0.2
    
    opt.dataset_mode='multi_snet_code'
    opt.vq_model='pvqvae'
    opt.vq_dset='snet'
    opt.vq_cat='all'

    # pix3d
    # opt.dataset_mode='pix3d_img'
    # opt.vq_dset='pix3d_code'
    # opt.cat = 'chair'
    # opt.cat = 'all'
    # opt.pix3d_mode='noBG'

    train_dataset, test_dataset = CreateDataset(opt)

    test_dl = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=int(opt.nThreads))


    test_dg = get_data_generator(test_dl)
    
    return test_dl, test_dg
    

def get_textdf_dset(opt):
    ##### setup dataset
    opt.ratio = 1.0

    # ShapeNet
    opt.cat = 'chair'
    opt.trunc_thres = 0.2
    
    opt.dataset_mode='shapenet_lang'
    opt.vq_model='pvqvae'
    opt.vq_dset='snet'
    opt.vq_cat='all'

    train_dataset, test_dataset = CreateDataset(opt)

    test_dl = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=int(opt.nThreads))


    test_dg = get_data_generator(test_dl)
    
    return test_dl, test_dg

def get_textdf_model(opt, ckpt=''):
    
    # load df stuff
    opt.model='textdf'
    opt.tf_cfg='configs/df_snet_code.yaml'
  
    # load vq stuff
    opt.vq_model='pvqvae'
    opt.vq_cfg='configs/pvqvae_snet.yaml'
    opt.vq_ckpt='saved_ckpt/pretrained-pvqvae.pth'
    
    opt.vq_dset='snet'
    opt.vq_note = 'default'

    if ckpt != '':
        opt.ckpt = ckpt
    model = create_model(opt)
    print(f'[*] "{opt.model}" initialized.')
    model.load_ckpt(opt.ckpt)
        
    return model


def get_imgdf_model(opt, ckpt=''):
    
    # load df stuff
    opt.model='imgdf'
    opt.tf_cfg='configs/df_snet_code.yaml'
  
    # load vq stuff
    opt.vq_model='pvqvae'
    opt.vq_cfg='configs/pvqvae_snet.yaml'
    opt.vq_ckpt='saved_ckpt/pretrained-pvqvae.pth'
    
    opt.vq_dset='snet'
    opt.vq_note = 'default'

    if ckpt != '':
        opt.ckpt = ckpt
    model = create_model(opt)
    print(f'[*] "{opt.model}" initialized.')
    model.load_ckpt(opt.ckpt)
        
    return model


def get_shape_comp_model(opt, ckpt=''):
    
    # load tf stuff
    opt.model='simple_multi_df'
    opt.tf_cfg='configs/df_snet_code.yaml'
    # load vq stuff
    opt.vq_model='pvqvae'
    opt.vq_cfg='configs/pvqvae_snet.yaml'
    opt.vq_ckpt='saved_ckpt/pretrained-pvqvae.pth'
    
    ### opt.vq_dset='sdf_code' # original
    opt.vq_dset='snet'
    opt.vq_note = 'default'

    if ckpt != '':
        opt.ckpt = ckpt
    model = create_model(opt)
    print(f'[*] "{opt.model}" initialized.')
    model.load_ckpt(opt.ckpt)
        
    return model

def get_pix3d_img_dset(opt):
    
    
    pix3d_img_dset = None
    
    return pix3d_img_dset

def get_resnet2vq_model(opt):
    resnet2vq_net = None
    
    return resnet2vq_net

def make_dummy_batch(bs):
    batch = {}
    batch['sdf'] = torch.zeros((bs,64,64,64))
    batch['idx'] = torch.zeros((bs,8,8,8),dtype=torch.int)
    batch['z_q'] = torch.zeros((bs,256,8,8,8))
    return batch

def preprocess_sdf(sdf):
    # chair legs
    theta = 0 * np.pi / 180.
    sx, sy, sz = 1, 1, 1
    tx, ty, tz = 0.0, 0.3, 0.05

    theta = torch.tensor([
        [np.cos(theta), 0, np.sin(theta), tx],
        [0., 1, 0, ty],
        [-np.sin(theta), 0, np.cos(theta), tz],
    ]).unsqueeze(0)
    size = sdf.shape
    affine_grid = torch.nn.functional.affine_grid(theta, size).to(sdf)
    ret = torch.nn.functional.grid_sample(sdf, affine_grid, mode='bilinear', padding_mode='border')

    return ret

def shape_comp(model, sdf, mode='bottom', n_gen=16, topk=30):
    from models.pvqvae_model import PVQVAEModel
    
    device = sdf.device

    # given sdf, figure out what is known
    sdf_partial = sdf.clone()
    sdf_partial = sdf_partial.clamp(-0.2, 0.2)

    gen_order = torch.arange(512).cuda()

    # extract code with pvqvae
    cur_bs = sdf_partial.shape[0]
    sdf_partial_cubes = PVQVAEModel.unfold_to_cubes(sdf_partial).to(device)

    zq_cubes, _, info = model.vqvae.encode(sdf_partial_cubes)
    zq_voxels = PVQVAEModel.fold_to_voxels(zq_cubes, batch_size=cur_bs, ncubes_per_dim=8)
    
    quant = zq_voxels
    _, _, quant_ix = info
    d, h, w = quant.shape[-3:]
    # quant_ix = rearrange(quant_ix, '(b d h w) -> b d h w', b=cur_bs, d=d, h=h, w=w)

    # uni, cnts = quant_ix.unique(return_counts=True)
    # token_to_cnt = {uni[i].item(): cnts[i].item() for i in range(len(uni))}
    # print(token_to_cnt)

    gen_order = gen_order.view(8, 8, 8)
    gen_order[:, 4:, :] = -1.
    gen_order = gen_order[gen_order != -1].view(-1)

    d, h, w = quant.shape[-3:]
    quant_ix = rearrange(quant_ix, '(b d h w) -> b d h w', b=cur_bs, d=d, h=h, w=w)

    comp_data = {}
    comp_data['sdf'] = sdf_partial.cpu()
    comp_data['idx'] = quant_ix.cpu()
    comp_data['z_q'] = quant.cpu()
    
    # B = 16
    comp_data = make_batch(comp_data, B=n_gen)

    # model.inference(comp_data, gen_order=gen_order, topk=30)
    model.inference(comp_data, gen_order=gen_order, topk=topk)
    
    return model.x_recon_tf