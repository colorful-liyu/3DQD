import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from tarfile import RECORDSIZE
import numpy as np
from tqdm import trange

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import pytorch3d
import pytorch3d.io
import time

from utils.demo_util import get_shape_comp_opt, get_textdf_model, get_textdf_dset
from metrics.textdf_metrics import compute_text_metrics

from utils.util import get_logger
from tqdm.auto import tqdm
from tqdm import trange
import argparse
import copy

def sample_sdf_from_set(model, test_dg, gen_num, gen_bs, guidew):
    logger.info('Sampling')
    gen_list = []
    text_list = []
    gt_list = []
    for i, data_orin in tqdm(enumerate(test_dg), total=gen_num):
        if i == gen_num:
            break
        data = copy.deepcopy(data_orin)
        del data_orin  
        gen = model.uncond_gen(bs=gen_bs, guidew=guidew, class_label=1, text_cond=data['text']).cpu()
        gen_list.append(gen)
        text_list.append(data['text'][0])
        gt_list.append(data['sdf'].cpu())
    gen_list = torch.cat(gen_list, dim=0)
    gt_list = torch.cat(gt_list, dim=0)
    return gen_list, gt_list, text_list

def sample_sdf_from_txt(model, text_list, gen_num, gen_bs, guidew):
    logger.info('Sampling from txt')
    gen_list = []
    for i in trange(len(text_list)):
        gen = model.uncond_gen(bs=gen_bs, guidew=guidew, class_label=1, text_cond=[text_list[i]]).cpu()
        gen_list.append(gen)
    gen_list = torch.cat(gen_list, dim=0)
    return gen_list


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--seed', type=int, default=111)
parser.add_argument('--gen_num', type=int, default=500)
parser.add_argument('--num_points', type=int, default=2048)
parser.add_argument('--gen_bs', type=int, default=9)
parser.add_argument('--classifier_free', type=float, default=0.0)
parser.add_argument('--guidew', type=float, default=0.0)

parser.add_argument('--load_gen_path', type=str, default='')
parser.add_argument('--cat', type=str, default='chair')
parser.add_argument('--note', type=str, default='')
parser.add_argument('--ckpt', type=str, default='saved_ckpt/textdf-all-LR1e-4-lrdecay15_dual_resadd_cf0.pth')
parser.add_argument('--EMD_flag', type=bool, default=False)
parser.add_argument('--save_flag', type=bool, default=False)
parser.add_argument('--regen_from_txt', type=bool, default=False)
parser.add_argument('--mode', type=str, default='l2norm', choices=['None', 'shape_unit', 'shape_bbox', 'l2norm'])

args = parser.parse_args()
""" setup opt"""

opt = get_shape_comp_opt(gpu_id=args.gpu_id, seed = args.seed)

opt.classifier_free=args.classifier_free
opt.guidew=args.guidew
opt.mode=args.mode     # [None, 'shape_unit', 'shape_bbox']
opt.load_gen_path=args.load_gen_path
opt.gen_num = args.gen_num
opt.num_points = args.num_points
opt.EMD_flag = args.EMD_flag
opt.save_flag = args.save_flag
opt.gen_bs = args.gen_bs
opt.regen_from_txt = args.regen_from_txt
all_cats=[args.cat]
test_dl, test_dg = get_textdf_dset(opt)

""" setup different model """
model = get_textdf_model(opt, ckpt = args.ckpt)  
model.eval()

""" setup dir """
opt.save_dir=f'cameraready/text_samples/{all_cats[0]}_{opt.gen_num}_{args.note}_{opt.mode}_level0.04_{int(time.time())}'
if not os.path.exists(opt.save_dir): 
    os.makedirs(opt.save_dir)

logger = get_logger('test', opt.save_dir)

for k, v in vars(opt).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))


with torch.no_grad():

    if opt.load_gen_path != '':
        gt_list = torch.tensor(np.load(f'{opt.load_gen_path}/gt.npy'))
        gen_list = torch.tensor(np.load(f'{opt.load_gen_path}/gen.npy')).reshape(len(gt_list), -1, 1, 64, 64, 64)
        text_list=[]
        with open(f'{opt.load_gen_path}/text.txt') as f:
            for line in f.readlines():
                line = line.strip()
                text_list.append(line)
        text_list = text_list[0:opt.gen_num]
        gt_list = gt_list[0:opt.gen_num]
        gen_list = gen_list[0:opt.gen_num].reshape(-1, 1, 64, 64, 64)
        if opt.regen_from_txt:
            gen_list = sample_sdf_from_txt(model, text_list, opt.gen_num, opt.gen_bs, opt.guidew)
    else:
        gen_list, gt_list, text_list = sample_sdf_from_set(model, test_dg, opt.gen_num, opt.gen_bs, opt.guidew)


    # save
    fw = open(os.path.join(opt.save_dir, 'text.txt'), 'w')
    for i in range(len(text_list)):
        fw.write(text_list[i])
        fw.write('\n')
    fw.close
    np.save(os.path.join(opt.save_dir, 'gen.npy'), gen_list.numpy())
    np.save(os.path.join(opt.save_dir, 'gt.npy'), gt_list.numpy())
    
    logger.info('Computing metrics')
    results = compute_text_metrics(opt.device, gen_list, gt_list, text_list, opt.gen_bs, opt.mode, logger, opt.EMD_flag, num_points=opt.num_points, save_flag=opt.save_flag, save_dir = opt.save_dir)

for k, v in results.items():
    logger.info('%s: %.12f' % (k, v))
print(results)