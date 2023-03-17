import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from tarfile import RECORDSIZE
import imageio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import trange
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
cudnn.benchmark = True

import pytorch3d
import pytorch3d.io
import json

# some utility function for visualization
import utils
from utils.util_3d import init_mesh_renderer, sdf_to_mesh, add_mesh_textures

# some utility function
from utils.qual_util import save_mesh_as_gif
from IPython.display import Image as ipy_image
from IPython.display import display

from utils.demo_util import get_shape_comp_opt
from utils.demo_util import get_shape_comp_model
from metrics.shape_comp_metrics import compute_all_shape_comp_metrics

import utils
from utils.qual_util import get_partial_shape_by_range
from preprocess.process_one_mesh import process_obj
from utils.util import get_logger

def generate_from_partial_input(device, model_list, class_list, gen_bs = 9, guidew = 0, comp_type='half', fr=0.5):
    
    # range: -1 ~ 1.
    # x: left-to-right; y: bottom-to-top; z: front-to-back
    # example: only conditioning on the bottom of the chair
    if comp_type == 'half':
        min_x, max_x = -1., 1.
        min_y, max_y = -1., 0.
        min_z, max_z = -1., 1.
    elif comp_type == 'octant':
        min_x, max_x = -1., 0.
        min_y, max_y = -1., 0.
        min_z, max_z = -1., 0.
    input_range = {'x1': min_x, 'x2': max_x, 'y1': min_y, 'y2': max_y, 'z1': min_z, 'z2': max_z}

    logger.info('Sampling from partial shape')
    gt_sdf = []
    comp_sdf = []
    for i in trange(len(model_list)):
        class_label = class_list[i]
        sdf = utils.util_3d.read_sdf(model_list[i]).to(device)
        with torch.no_grad():
            """ perform shape completion """
            shape_comp_input = get_partial_shape_by_range(sdf, input_range, [class_label])
            _, comp_output = model.shape_comp(shape_comp_input, input_range, bs=gen_bs, topk=30, guidew=guidew, fr=fr)
            gt_sdf.append(sdf.float())
            comp_sdf.append(comp_output.float())
    return gt_sdf, comp_sdf


def load_shape_dataset(max_dataset_size):
    with open(f'{args.dataroot}/ShapeNet/info.json') as f:
            info = json.load(f)
            
            cat_to_id = info['cats']
            id_to_cat = {v: k for k, v in cat_to_id.items()}
            all_cats = info['all_cats']

            i = 1
            model_list = []
            cats_list = []
            class_list = []
            for c in all_cats:
                synset = info['cats'][c]
                with open(f'{args.dataroot}/ShapeNet/filelists/{synset}_test.lst') as f:  
                    model_list_s = []

                    tmp_pcs=[]
                    for l in f.readlines():
                        model_id = l.rstrip('\n')      
                        path = f'{args.dataroot}/ShapeNet/SDF_v1_64/{synset}/{model_id}/ori_sample.h5'
                        model_list_s.append(path)

                    model_list += model_list_s
                    cats_list += [synset] * len(model_list_s)
                    class_list += [i] * len(model_list_s)
                    i += 1
                    print('[*] %d samples for %s (%s).' % (len(model_list_s), id_to_cat[synset], synset))

            np.random.default_rng(seed=0).shuffle(model_list)
            np.random.default_rng(seed=0).shuffle(cats_list)
            np.random.default_rng(seed=0).shuffle(class_list)
            model_list = model_list[:max_dataset_size]
            class_list = class_list[:max_dataset_size]
            print('[*] %d samples loaded.' % (len(model_list)))

    return model_list, class_list

import argparse
import time

""" setup opt"""


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--guidew', type=float, default=0.5)
parser.add_argument('--cf', type=float, default=0.5)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--num', type=int, default=500)
parser.add_argument('--comp_type', type=str, default='half')
parser.add_argument('--dataroot', type=str, default='../dataset')
parser.add_argument('--ckpt', type=str, default='saved_ckpt/pretrained-shape-comp-cf0.5-epoch200.pth')
parser.add_argument('--gen_bs', type=int, default=10)
parser.add_argument('--note', type=str, default='')
parser.add_argument('--load_gen_path', type=str, default='')  # load generated results

args = parser.parse_args()

gpu_id = args.gpu_id
opt = get_shape_comp_opt(gpu_id=gpu_id)
opt.classifier_free=args.cf
opt.guidew=args.guidew
max_dataset_size = args.num
opt.comp_type = args.comp_type
opt.gen_bs = args.gen_bs
opt.load_gen_path = args.load_gen_path

""" setup different model """
model = get_shape_comp_model(opt, ckpt=args.ckpt)  
# model = get_shape_comp_model(opt, ckpt='saved_ckpt/rand_tf-snet_code-all-LR1e-4-clean-epoch200.pth')  
# model = get_shape_comp_model(opt, ckpt='logs/simple_multi_df-snet_code-all-LR1e-4-3multi-coarsemlp-resadd/ckpt/df_epoch-45.pth')  
model.eval()

""" setup dir """
res_dir = f'cameraready/shape_comp_samples/reorg/{args.num}_{opt.guidew}_{opt.comp_type}_{int(time.time())}'
if not os.path.exists(res_dir): 
    os.makedirs(res_dir)

logger = get_logger('test', res_dir)

for k, v in vars(opt).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

""" setup renderer """
dist, elev, azim = 1.7, 20, 110
mesh_renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=opt.device)


"""
    Shape completion 2 - structured input
    - given a mesh, we first extract SDF from that mesh.
    - then user can specify the partial input by setting the min & max values of x, y, z.
"""
model_list, class_list = load_shape_dataset(max_dataset_size)

torch.cuda.set_device(opt.device)


# for fr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
for fr in [0.5]:
    logger.info('[FR::%s] ' % (fr))
    
    if opt.load_gen_path == '':
        gt_sdf, comp_sdf = generate_from_partial_input(opt.device, model_list, class_list, gen_bs = opt.gen_bs, guidew = opt.guidew, comp_type=opt.comp_type, fr=fr)
        gen_list = torch.stack(comp_sdf, dim = 0).reshape(len(gt_sdf),-1, 1, 64, 64, 64).cpu().numpy()
        gt_list = torch.stack(gt_sdf, dim = 0).reshape(len(gt_sdf),-1, 1, 64, 64, 64).cpu().numpy()
        class_label_list = np.array(class_list)
        np.save(os.path.join(res_dir, 'all_gen.npy'), gen_list)
        np.save(os.path.join(res_dir, 'all_gt.npy'), gt_list)
        np.save(os.path.join(res_dir, 'all_label.npy'), class_label_list)
    else:
        comp_sdf = torch.tensor(np.load(f'{opt.load_gen_path}/all_gen.npy')).to(opt.device).reshape(max_dataset_size,-1, 1, 64, 64, 64)
        gt_sdf = torch.tensor(np.load(f'{opt.load_gen_path}/all_gt.npy')).to(opt.device).reshape(max_dataset_size,-1, 1, 64, 64, 64)

    record = compute_all_shape_comp_metrics(gt_sdf, comp_sdf, opt.comp_type, 'l2norm')
    

    for k, v in record.items():
        logger.info('%s: %.12f' % (k, v))

