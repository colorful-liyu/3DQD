import os
from collections import OrderedDict

import numpy as np
import einops
# import marching_cubes as mcubes
import mcubes
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.pvqvae_networks.auto_encoder import PVQVAE

import utils.util
from utils.util import instantiate_from_config, load_yaml_config
from utils.util_3d import init_mesh_renderer, render_sdf

class DiffusionModel(BaseModel):
    def name(self):
        return 'Diffusion-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.classifier_free = opt.classifier_free
        self.guidew = opt.guidew
        self.opt = opt
        # -------------------------------
        # Define Networks
        # -------------------------------

        assert opt.vq_cfg is not None

        # load configs for tf and vq
        tf_conf = OmegaConf.load(opt.tf_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)

        # init vqvae for decoding shapes
        mparam = vq_conf.model.params
        n_embed = mparam.n_embed
        embed_dim = mparam.embed_dim
        ddconfig = mparam.ddconfig

        n_down = len(ddconfig.ch_mult) - 1

        self.vqvae = PVQVAE(ddconfig, n_embed, embed_dim)
        self.load_vqvae(opt.vq_ckpt)
        self.vqvae.to(opt.device)
        self.vqvae.eval()
        
        # TO DO LIST
        if opt.model == 'df':
            self.df_config = load_yaml_config('configs/vq.yaml')
        elif opt.model ==  'simple_multi_df':
            self.df_config = load_yaml_config('configs/simple_multi_vq.yaml')
        elif opt.model == 'imgdf':
            self.df_config = load_yaml_config('configs/img_vq.yaml')
        elif opt.model == 'textdf':
            self.df_config = load_yaml_config('configs/text_vq.yaml')
            self.text_codec_config = load_yaml_config('configs/tokenizer.yaml')
            self.condition_codec = instantiate_from_config(self.text_codec_config['model'])
            self.condition_codec.to(opt.device)
        self.df = instantiate_from_config(self.df_config['model'])
        self.df.to(opt.device)


        if self.isTrain:
            # ----------------------------------
            # define loss functions
            # ----------------------------------
            # self.criterion_nll = nn.NLLLoss()
            self.criterion_ce = nn.CrossEntropyLoss()
            self.criterion_ce.to(opt.device)

            # ---------------------------------
            # initialize optimizers
            # ---------------------------------
 
            self.optimizer = optim.Adam([p for p in self.df.parameters() if p.requires_grad == True], lr=opt.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 30, 0.9)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        # transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        resolution = 64
        self.resolution = resolution

        # start token
        self.sos = 0
        self.counter = 0

        # init grid for lookup
        pe_conf = tf_conf.pe
        self.grid_size = pe_conf.zq_dim
        self.grid_table = self.init_grid(pos_dim=pe_conf.pos_dim, zq_dim=self.grid_size)

        # setup hyper-params 
        nC = resolution
        self.cube_size = 2 ** n_down # patch_size
        self.stride = self.cube_size
        self.ncubes_per_dim = nC // self.cube_size
        assert nC == 64, 'right now, only trained with sdf resolution = 64'
        assert (nC % self.cube_size) == 0, 'nC should be divisable by cube_size'

        # setup renderer
        dist, elev, azim = 1.7, 20, 20   
        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.opt.device)
        
    def load_vqvae(self, vq_ckpt):
        assert type(vq_ckpt) == str         
        state_dict = torch.load(vq_ckpt)

        self.vqvae.load_state_dict(state_dict)
        print(colored('[*] VQVAE: weight successfully load from: %s' % vq_ckpt, 'blue'))

    def init_grid(self, pos_dim=3, zq_dim=8):
        x = torch.linspace(-1, 1, zq_dim)
        y = torch.linspace(-1, 1, zq_dim)
        if pos_dim == 3:
            z = torch.linspace(-1, 1, zq_dim)
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
            grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
            pos_sos = torch.tensor([-1., -1., -1-2/zq_dim]).float().unsqueeze(0)
        else:
            grid_x, grid_y = torch.meshgrid(x, y)
            grid = torch.stack([grid_x, grid_y], dim=-1)
            pos_sos = torch.tensor([-1., -1-2/zq_dim]).float().unsqueeze(0)

        grid_table = grid.view(-1, pos_dim)
        grid_table = torch.cat([pos_sos, grid_table], dim=0)
        return grid_table

    def get_gen_order(self, sz, device):
        # return torch.randperm(sz).to(device)
        return torch.randperm(sz, device=device)
        # return torch.arange(sz).to(device)

    def get_dummy_input(self, bs=1, class_label=0, vq_init=None):
        
        ret = {}
        ret['sdf'] = torch.zeros(bs, 1, 64, 64, 64)#.to(device)
        if vq_init == None:
            ret['idx'] = torch.ones(bs, self.grid_size, self.grid_size, self.grid_size).long()*512#.to(device)
        else:
            ret['idx'] = vq_init
        ret['z_q'] = torch.zeros(bs, 256, self.grid_size, self.grid_size, self.grid_size)#.to(device)
        if class_label==0:
            ret['class_label'] = torch.randint(1,14,(bs,))
        else:
            ret['class_label'] = torch.ones(bs)*class_label
        if self.opt.model == 'textdf':
            ret['text'] = ['']*bs
        else:
            ret['text'] = None
        
        return ret


    def set_input(self, input=None, gen_order=None, text_cond=None, img_cond=None):
        
        self.x = input['sdf']
        self.x_idx = input['idx']
        self.z_q = input['z_q']
        self.class_label = input['class_label']    

        if self.opt.model == 'textdf':
            cond = self.condition_codec.get_tokens(text_cond if text_cond else input['text'])
            self.text = cond['token']
            self.text_mask = cond['mask']
        else:
            self.text = None
            self.text_mask = None
        if self.opt.model == 'imgdf':
            self.img = img_cond if img_cond!=None else input['img_logits']
        else:
            self.img = None

        bs, dz, hz, wz = self.x_idx.shape
        self.z_shape = self.z_q.shape

        if self.opt.dataset_mode in ['pix3d_img', 'snet_img']:
            self.gt_vox = input['gt_vox']

        self.x_idx_seq = rearrange(self.x_idx, 'bs dz hz wz -> (dz hz wz) bs').contiguous() # to (T, B)
        self.x_idx = self.x_idx_seq.clone()

        # prepare input for transformer
        T, B = self.x_idx.shape[:2]
        
        if gen_order is None:
            self.gen_order = self.get_gen_order(T, self.opt.device)
            self.context_len = -1 # will be specified in inference
        else:
            if len(gen_order) != T:
                
                self.context_len = len(gen_order)
                # pad the remaining
                remain = torch.tensor([i for i in range(T) if i not in gen_order]).to(gen_order)
                remain = remain[torch.randperm(len(remain))]
                self.gen_order = torch.cat([gen_order, remain])
            else:
                self.gen_order = gen_order

        x_idx_seq_shuf = self.x_idx_seq[self.gen_order]
        x_seq_shuffled = torch.cat([torch.LongTensor(1, bs).fill_(self.sos), x_idx_seq_shuf], dim=0)  # T+1
        pos_shuffled = torch.cat([self.grid_table[:1], self.grid_table[1:][self.gen_order]], dim=0)   # T+1, <sos> should always at start.

        self.inp = x_seq_shuffled[:-1].clone()
        self.tgt = x_seq_shuffled[1:].clone()
        self.inp_pos = pos_shuffled[:-1].clone()
        self.tgt_pos = pos_shuffled[1:].clone()

        self.counter += 1

        vars_list = ['gen_order',
                     'inp', 'inp_pos', 'tgt', 'tgt_pos',
                     'x_idx', 'x_idx_seq', 'z_q', 'x', 'class_label']
        if self.opt.model == 'textdf':
            vars_list.append('text')
        if self.opt.model == 'imgdf':
            vars_list.append('img')

        self.tocuda(var_names=vars_list)

    def forward(self):
        """ given 
                inp, inp_pos, tgt_pos
            infer
                tgt
            outp is the prob. dist. over x_(t+1) at pos_(t+1)
            p(x_{t+1} | x_t, pos_t, pos_{t+1})
        """

        context_mask = torch.bernoulli(torch.ones_like(self.class_label)-self.classifier_free).to(self.class_label.device)
        class_label = self.class_label * context_mask
        
        self.outp = self.df(self.x_idx_seq.permute(1, 0), class_label=class_label, text=self.text, img=self.img)#[:-1]


    @torch.no_grad()
    def generate_content(
        self,
        *,
        batch,
        condition=None,
        filter_ratio = 0.5,
        temperature = 1.0,
        content_ratio = 0.0,
        replicate=1,
        return_att_weight=False,
        sample_type="top0.85r",
    ):
        self.eval()
        if condition is None:
            condition = self.prepare_condition(batch=batch)
        else:
            condition = self.prepare_condition(batch=None, condition=condition)
        
        batch_size = len(batch['text']) * replicate

        if self.learnable_cf:
            cf_cond_emb = self.transformer.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            batch['text'] = [''] * batch_size
            cf_condition = self.prepare_condition(batch=batch)
            cf_cond_emb = self.transformer.condition_emb(cf_condition['condition_token']).float()
        
        def cf_predict_start(log_x_t, cond_emb, t):
            log_x_recon = self.transformer.predict_start(log_x_t, cond_emb, t)[:, :-1]
            if abs(self.guidance_scale - 1) < 1e-3:
                return torch.cat((log_x_recon, self.transformer.zero_vector), dim=1)
            cf_log_x_recon = self.transformer.predict_start(log_x_t, cf_cond_emb.type_as(cond_emb), t)[:, :-1]
            log_new_x_recon = cf_log_x_recon + self.guidance_scale * (log_x_recon - cf_log_x_recon)
            log_new_x_recon -= torch.logsumexp(log_new_x_recon, dim=1, keepdim=True)
            log_new_x_recon = log_new_x_recon.clamp(-70, 0)
            log_pred = torch.cat((log_new_x_recon, self.transformer.zero_vector), dim=1)
            return log_pred

        if replicate != 1:
            for k in condition.keys():
                if condition[k] is not None:
                    condition[k] = torch.cat([condition[k] for _ in range(replicate)], dim=0)
            
        content_token = None

        if len(sample_type.split(',')) > 1:
            if sample_type.split(',')[1][:1]=='q':
                self.transformer.p_sample = self.p_sample_with_truncation(self.transformer.p_sample, sample_type.split(',')[1])
        if sample_type.split(',')[0][:3] == "top" and self.truncation_forward == False:
            self.transformer.cf_predict_start = self.predict_start_with_truncation(cf_predict_start, sample_type.split(',')[0])
            self.truncation_forward = True

        if len(sample_type.split(',')) == 2 and sample_type.split(',')[1][:4]=='time' and int(float(sample_type.split(',')[1][4:])) >= 2:
            trans_out = self.transformer.sample_fast(condition_token=condition['condition_token'],
                                                condition_mask=condition.get('condition_mask', None),
                                                condition_embed=condition.get('condition_embed_token', None),
                                                content_token=content_token,
                                                filter_ratio=filter_ratio,
                                                temperature=temperature,
                                                return_att_weight=return_att_weight,
                                                return_logits=False,
                                                print_log=False,
                                                sample_type=sample_type,
                                                skip_step=int(float(sample_type.split(',')[1][4:])-1))

        else:
            if 'time' in sample_type and float(sample_type.split(',')[1][4:]) < 1:
                self.transformer.prior_ps = int(1024 // self.transformer.num_timesteps * float(sample_type.split(',')[1][4:]))
                if self.transformer.prior_rule == 0:
                    self.transformer.prior_rule = 1
                self.transformer.update_n_sample()
            trans_out = self.transformer.sample(condition_token=condition['condition_token'],
                                            condition_mask=condition.get('condition_mask', None),
                                            condition_embed=condition.get('condition_embed_token', None),
                                            content_token=content_token,
                                            filter_ratio=filter_ratio,
                                            temperature=temperature,
                                            return_att_weight=return_att_weight,
                                            return_logits=False,
                                            print_log=False,
                                            sample_type=sample_type)


        content = self.content_codec.decode(trans_out['content_token'])  #(8,1024)->(8,3,256,256)
        self.train()
        out = {
            'content': content
        }
        

        return out


    @torch.no_grad()
    def inference(
        self,
        data,
        clip = None,
        temperature = 1.,
        return_rec = True,
        filter_ratio = [0.5],
        content_ratio = [1], # the ratio to keep the encoded content tokens
        return_att_weight=False,
        return_logits=False,
        sample_type="normal",
        guidew=0,
        text_cond=None,
        img_cond=None,
        seq_len=None, gen_order=None, topk=None, prob=None, alpha=1., should_render=False, verbose=False,
        **kwargs,
    ):
        self.df.eval()
        if guidew == None:
            guidew = 0


        # context: 
        #     - if prob is given, seq_len=1
        #     - else seq_len is defined by gen_order
        if prob is not None:
            if seq_len is None:
                seq_len = 1 # context
        else:
            if gen_order is None:
                if seq_len is None:
                    seq_len = 1 # context
            else:
                # if goes here, context_len will be given by gen_order
                # +1 to include sos
                seq_len = len(gen_order)+1

        self.set_input(data, gen_order=gen_order, text_cond=text_cond, img_cond=img_cond)

        T = self.x_idx_seq.shape[0] # +1 since <sos>
        B = self.x_idx_seq.shape[1]


        for fr in filter_ratio:
            for cr in content_ratio:
                num_content_tokens = int(T * cr)
                
                trans_out = self.df.sample(condition_token=self.img if self.opt.model == 'imgdf' else self.text,
                                                    condition_mask=None,
                                                    condition_embed=None,
                                                    class_label=self.class_label,
                                                    content_token=self.x_idx_seq[:num_content_tokens].permute(1, 0) if num_content_tokens>=0 else self.x_idx_seq.permute(1, 0),
                                                    filter_ratio=fr,
                                                    temperature=temperature,
                                                    return_att_weight=return_att_weight,
                                                    return_logits=return_logits,
                                                    # content_logits=content.get('content_logits', None),
                                                    sample_type=sample_type,
                                                    batch_size=B,
                                                    prob=prob,
                                                    guidew=guidew,
                                                    **kwargs)

        self.df.train() 
        self.x_recon_df = self.vqvae.decode_enc_idices(trans_out['content_token'].permute(1, 0), z_spatial_dim=self.grid_size)

        return trans_out['content_token'].permute(1, 0)

    # application func
    def uncond_gen(self, bs=1, topk=30, guidew=0, class_label=0, text_cond=None):
        if text_cond:
            text_cond = text_cond * bs
        
        # get dummy data
        data = self.get_dummy_input(bs=bs, class_label=class_label)
        self.inference(data, seq_len=None, topk=topk, filter_ratio=[1.0], guidew=guidew, text_cond=text_cond)

        gen_df = self.x_recon_df
        return gen_df

    # application func
    def text_edit(self, bs=1, topk=30, guidew=0, text_cond=None, former_shape_index=None, fr=1.0):
        if text_cond:
            text_cond = text_cond * bs
        # get dummy data
        data = self.get_dummy_input(bs=bs, class_label=1, vq_init = former_shape_index)
        shape_index = self.inference(data, seq_len=None, topk=topk, filter_ratio=[fr], guidew=guidew, text_cond=text_cond)
        shape_index = rearrange(shape_index, '(d h w) bs -> bs d h w ', h=8, w=8, d=8)

        gen_df = self.x_recon_df
        return gen_df, shape_index

    def shape_comp(self, input, input_range, bs=6, topk=30, guidew=0, fr=0.5):
        from models.pvqvae_model import PVQVAEModel
        from utils.qual_util import make_batch, get_shape_comp_input_mesh

        min_x, max_x = input_range['x1'], input_range['x2']
        min_y, max_y = input_range['y1'], input_range['y2']
        min_z, max_z = input_range['z1'], input_range['z2']
        
        bins_x = np.linspace(-1, 1, num=9)
        bins_y = np.linspace(-1, 1, num=9)
        bins_z = np.linspace(-1, 1, num=9)


        # -1: 1, 1: 9
        # find cube idx
        x_inds = np.digitize([min_x, max_x], bins_x)
        y_inds = np.digitize([min_y, max_y], bins_y)
        z_inds = np.digitize([min_z, max_z], bins_z)

        x_inds -= 1
        y_inds -= 1
        z_inds -= 1

        # [0 8] [0 4] [0 8]
        cube_x1, cube_x2 = x_inds
        cube_y1, cube_y2 = y_inds
        cube_z1, cube_z2 = z_inds

        # first obtain tokens from input
        sdf_partial, sdf_missing, gen_order, class_label = input['sdf'], input['sdf_missing'], input['gen_order'], input['class_label']
        # sdf_partial, sdf_missing, gen_order = shape_comp_input['sdf'], shape_comp_input['sdf_missing'], shape_comp_input['gen_order']

        # extract code with pvqvae
        cur_bs = sdf_partial.shape[0]
        sdf_partial_cubes = PVQVAEModel.unfold_to_cubes(sdf_partial).to(self.opt.device)

        zq_cubes, _, info = self.vqvae.encode(sdf_partial_cubes)
        zq_voxels = PVQVAEModel.fold_to_voxels(zq_cubes, batch_size=cur_bs, ncubes_per_dim=8)
        quant = zq_voxels
        _, _, quant_ix = info
        d, h, w = quant.shape[-3:]

        quant_ix = rearrange(quant_ix, '(b d h w) -> b d h w', b=cur_bs, d=d, h=h, w=w)
        # quant_ix[:, 0:8, 4:8, 0:8] = 512
        quant_ix[:, :cube_x1, :, :] = 512
        quant_ix[:, cube_x2:, :, :] = 512
        quant_ix[:, :, :cube_y1, :] = 512
        quant_ix[:, :, cube_y2:, :] = 512
        quant_ix[:, :, :, :cube_z1] = 512
        quant_ix[:, :, :, cube_z2:] = 512
        # quant_ix[:, cube_x1:cube_x2, cube_y1:cube_y2, cube_z1:cube_z2] = 512

        comp_data = {}
        comp_data['sdf'] = sdf_partial.cpu()
        comp_data['idx'] = quant_ix.cpu()
        comp_data['z_q'] = quant.cpu()
        comp_data['sdf_res'] = sdf_missing.cpu()
        comp_data['class_label'] = torch.tensor(class_label)
        comp_data = make_batch(comp_data, B=bs)

        self.inference(comp_data, gen_order=gen_order, topk=topk,  filter_ratio=[fr], guidew=guidew)

        input_mesh = 0
        # input_mesh = get_shape_comp_input_mesh(comp_data['sdf'], comp_data['sdf_res'])
        # input_mesh = input_mesh.to(self.x_recon_df)

        return input_mesh, self.x_recon_df


    def denoise_experiment(self, input, bs=6, topk=30, guidew=0, filter_ratio=0.5):
        from models.pvqvae_model import PVQVAEModel
        from utils.qual_util import make_batch, get_shape_comp_input_mesh

        # first obtain tokens from input
        sdf, class_label = input['sdf'], input['class_label']


        # extract code with pvqvae
        cur_bs = sdf.shape[0]
        sdf_cubes = PVQVAEModel.unfold_to_cubes(sdf).to(self.opt.device)

        zq_cubes, _, info = self.vqvae.encode(sdf_cubes)
        zq_voxels = PVQVAEModel.fold_to_voxels(zq_cubes, batch_size=cur_bs, ncubes_per_dim=8)
        quant = zq_voxels
        _, _, quant_ix = info
        d, h, w = quant.shape[-3:]

        quant_ix = rearrange(quant_ix, '(b d h w) -> b d h w', b=cur_bs, d=d, h=h, w=w)

        comp_data = {}
        comp_data['sdf'] = sdf.cpu()
        comp_data['idx'] = quant_ix.cpu()
        comp_data['z_q'] = quant.cpu()
        comp_data['class_label'] = torch.tensor(class_label)
        comp_data = make_batch(comp_data, B=bs)

        self.inference(comp_data, gen_order=None, topk=topk,  filter_ratio=[filter_ratio], guidew=guidew)

        input_mesh = 0

        return input_mesh, self.x_recon_df
    

    def single_view_recon(self, img_tensor, resnet2vq, bs=1, make_diff=True):
        from utils.qual_util import get_img_prob
        
        # encode the img to vq
        if make_diff:
            img_tensor = img_tensor.repeat(bs, 1, 1, 1)
            img_logits = resnet2vq(img_tensor) # bs c d h w
            img_logits = F.softmax(img_logits, dim=1) # compute the prob. of next ele
            img_grid = img_logits.argmax(dim=1).cpu()
            img_logits = rearrange(img_grid, 'bs d h w -> bs (d h w)')
        else:
            img_logits = resnet2vq(img_tensor) # bs c d h w
            img_logits = F.softmax(img_logits, dim=1) # compute the prob. of next ele
            img_grid = img_logits.argmax(dim=1)
            img_grid = img_grid.repeat(bs, 1, 1, 1).cpu()
            img_logits = rearrange(img_grid, 'bs d h w -> bs (d h w)')
        # get dummy data
        data = self.get_dummy_input(bs=bs, class_label=1, vq_init=img_grid)
        # data = self.get_dummy_input(bs=bs, class_label=1)

        self.inference(data, filter_ratio=[0.6], guidew=0, img_cond=img_logits)

        return self.x_recon_df


    def get_transform_grids(self, B):
        Rt = repeat(self.Rt, 'b m n -> (repeat b) m n', repeat=B)
        S = repeat(self.S, 'b m n -> (repeat b) m n', repeat=B)

        device = self.opt.device
        gt_size = 32
        vmin, vmax = -1., 1.
        vrange = vmax - vmin
        x = torch.linspace(vmin, vmax, gt_size)
        y = torch.linspace(vmin, vmax, gt_size)
        z = torch.linspace(vmin, vmax, gt_size)
        xx, yy, zz = torch.meshgrid(x, y, z)

        grid_to_gt_res = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0).to(device)
        grid_to_gt_res = grid_to_gt_res.repeat(B, 1, 1, 1, 1)
        grid_affine = torch.nn.functional.affine_grid(Rt, (B, 1, 64, 64, 64)).to(device)
        grid_scale = torch.nn.functional.affine_grid(S, (B, 1, 64, 64, 64)).to(device)
        return grid_to_gt_res, grid_affine, grid_scale

    def eval_metrics(self, dataloader, thres=0.0):
        self.eval()
        
        ret = OrderedDict([
            ('iou', 0.0),
            ('iou_std', 0.0),
        ])
        self.train()
        return ret

    def backward(self):
        '''backward pass for the generator in training the unsupervised model'''
        # target = rearrange(self.tgt, 'seq b -> (seq b)')
        # outp = rearrange(self.outp, 'seq b cls-> (seq b) cls') # exclude the last one as its for <end>
        
        loss_nll = self.outp['loss']

        self.loss = loss_nll

        self.loss_nll = loss_nll
        self.loss.backward()

    def optimize_parameters(self, total_steps):
        # self.vqvae.train()

        self.set_requires_grad([self.df], requires_grad=True)

        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_logs_data(self):
        """ return a dictionary with
            key: graph name
            value: an OrderedDict with the data to plot
        
        """
        raise NotImplementedError
        return ret

    def get_current_errors(self):
        
        ret = OrderedDict([
            ('nll', self.loss_nll.data),
            # ('rec', self.loss_rec.data),
        ])

        return ret

    def get_current_visuals(self):

        with torch.no_grad():
            # self.image = render_sdf(self.renderer, self.x)
            # self.image_recon = render_sdf(self.renderer, self.x_recon)
            self.image_recon_df = render_sdf(self.renderer, self.x_recon_df)
            
        vis_tensor_names = [
            # 'image',
            # 'image_recon',
            'image_recon_df',
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        # vis_tensor_names = ['%s/%s' % (phase, n) for n in vis_tensor_names]
        visuals = zip(vis_tensor_names, vis_ims)
                            
        return OrderedDict(visuals)

    def save(self, label):
        
        state_dict = {
            'vqvae': self.vqvae.cpu().state_dict(),
            'df': self.df.cpu().state_dict(),
        }
        save_filename = 'df_%s.pth' % (label)
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(state_dict, save_path)
        self.vqvae.to(self.opt.device)
        self.df.to(self.opt.device)

    def load_ckpt(self, ckpt):
        if type(ckpt) == str:
            state_dict = torch.load(ckpt)
        else:
            state_dict = ckpt

        self.vqvae.load_state_dict(state_dict['vqvae'])
        self.df.load_state_dict(state_dict['df'])
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))
    
    def load_ckpt_from_text_to_img(self, ckpt):
        if type(ckpt) == str:
            state_dict = torch.load(ckpt)
        else:
            state_dict = ckpt
        ignore_keys = ['condition_emb']  # 'transformer.cond_emb' may be added
        self.vqvae.load_state_dict(state_dict['vqvae'])
        df_ckpt = state_dict['df']
        for k in list(df_ckpt.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    del df_ckpt[k]
        self.df.load_state_dict(df_ckpt, strict=False)
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))


    


