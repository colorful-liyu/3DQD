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
import time

# some utility function for visualization
import utils
from utils.util_3d import init_mesh_renderer, sdf_to_mesh, add_mesh_textures

# some utility function
from utils.qual_util import save_mesh_as_gif
from IPython.display import Image as ipy_image
from IPython.display import display

from utils.demo_util import get_shape_comp_opt
from utils.demo_util import get_shape_comp_model
from metrics.uncond_metrics import compute_all_metrics

from utils.util import get_logger
import utils
from utils.qual_util import get_partial_shape_by_range
from preprocess.process_one_mesh import process_obj
from tqdm.auto import tqdm
import argparse


def sample_sdf(model, gen_num, gen_bs, guidew, class_label, class_desc):

    iterator = range(0, gen_num, gen_bs)

    gen_list = []
    for b_start in tqdm(iterator, desc=class_desc):
        end = min(gen_num-b_start,  gen_bs)
        gen = model.uncond_gen(bs=end, guidew=guidew, class_label=class_label, text_cond=None).cpu()
        gen_list.append(gen)
    gen_list = torch.cat(gen_list, dim=0)

    return gen_list



@torch.no_grad()
def EVAL(opt, model, sdf_files, gen_num, gen_bs = 16, save_flag = False, guidew = 0, class_label=0, class_desc=''):
    if opt.load_gen_path != '':
        gen_list = torch.tensor(np.load(opt.load_gen_path))[:gen_num]
    else:
        gen_list = sample_sdf(model, gen_num, gen_bs, guidew, class_label, class_desc)
    if save_flag:
        np.save(os.path.join(opt.save_dir, 'sdf.npy'), gen_list.numpy())
    results = compute_all_metrics(opt.device, gen_list, sdf_files, 48, opt.mode, logger, opt.EMD_flag, num_points=opt.num_points, class_label=class_desc)
    return results

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=222) 
parser.add_argument('--num_points', type=int, default=2048)
parser.add_argument('--classifier_free', type=float, default=0.0)
parser.add_argument('--guidew', type=float, default=0.0)


# ../vqsdf/samples/chair_1311_E900_SEED333_EMD_None_level0.04_1666938028/sdf.npy
# ../vqsdf/samples/airplane_795_E1k_SEED222_EMD_None_level0.04_1667039000/sdf.npy
# ../vqsdf/samples/car_626_E1002_SEED222_EMD_None_level0.04_1667051539/sdf.npy

parser.add_argument('--load_gen_path', type=str, default='')
parser.add_argument('--cat', type=str, default='')
parser.add_argument('--note', type=str, default='')
parser.add_argument('--ckpt', type=str, default='')
parser.add_argument('--log_save_path', type=str, default='cameraready/uncond/reorg')
parser.add_argument('--dataroot', type=str, default='../dataset')
parser.add_argument('--EMD_flag', type=bool, default=True)
parser.add_argument('--mode', type=str, default='pf_norm', choices=['None', 'shape_unit', 'shape_bbox', 'pf_norm'])


args = parser.parse_args()
""" setup opt"""

opt = get_shape_comp_opt(gpu_id=args.gpu_id, seed=args.seed)
opt.classifier_free=args.classifier_free
opt.guidew=args.guidew
opt.seed=args.seed
opt.mode=args.mode
opt.load_gen_path=args.load_gen_path
opt.num_points = args.num_points
opt.EMD_flag = args.EMD_flag
all_cats=[args.cat]

""" setup different model """
model = get_shape_comp_model(opt, ckpt = args.ckpt)  
model.eval()

""" setup dir """
opt.save_dir=os.path.join(args.log_save_path, f'{all_cats[0]}_{args.note}_{opt.mode}_{int(time.time())}')
if not os.path.exists(opt.save_dir): 
    os.makedirs(opt.save_dir)

logger = get_logger('test', opt.save_dir)

for k, v in vars(opt).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))


logger.info('Loading datasets...')
with open(f'{args.dataroot}/ShapeNet/info.json') as f:
        info = json.load(f)
        
        cat_to_id = info['cats']
        id_to_cat = {v: k for k, v in cat_to_id.items()}

        i = 1
        model_list = []
        cats_list = []
        class_list = []
        for c in all_cats:
            synset = info['cats'][c]
            with open(f'{args.dataroot}/ShapeNet/filelists/{synset}_test.lst') as f:
                model_list_s = []
                for l in f.readlines():
                    model_id = l.rstrip('\n')
                    
                    path = f'{args.dataroot}/ShapeNet/SDF_v1_64/{synset}/{model_id}/ori_sample.h5'
                    model_list_s.append(path)
                    sdf = utils.util_3d.read_sdf(path)
                    model_list.append(sdf)
                cats_list += [synset] * len(model_list_s)
                class_list += [i] * len(model_list_s)
                i += 1
                print('[*] %d samples for %s (%s).' % (len(model_list_s), id_to_cat[synset], synset))

        np.random.default_rng(seed=0).shuffle(model_list)
        np.random.default_rng(seed=0).shuffle(cats_list)
        np.random.default_rng(seed=0).shuffle(class_list)
        print('[*] %d samples loaded.' % (len(model_list)))


all_cat_results=[]

for i in range(len(all_cats)):
    cats = all_cats[i] #取出对应类别的名字
    select_index = (np.array(class_list)==1) #取出对应类别的索引
    sdf_files = torch.cat(model_list, dim=0)[select_index] 
    results = EVAL(opt, model, sdf_files, gen_num=len(model_list), gen_bs = 32, save_flag = True, guidew = opt.guidew, class_label=1, class_desc=cats) # 795, 1311, 626
    all_cat_results.append(results)

for k, v in results.items():
    logger.info('%s: %.12f' % (k, v))
print(all_cat_results)