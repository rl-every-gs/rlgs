#
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import random 
import torch.nn.functional as F
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.gaussian_model_search_mem5 import SGaussianModelMem2  
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
import argparse
from arguments import ModelParams, PipelineParams, OptimizationParams
from helper_common import ATTRIBUTES1, ATTRIBUTES2, ATTRIBUTESALL, XYZROTATIONOPAICY
from helper_rl import try_gradient_updating, evaluate_total_gain_psnr, change_seed, try_gradient_updating_densi, get_psnr_dict, evaluate_total_gain_psnr_rt_list
from helper_tmgs import densify_iteration
import time
lr_search_space= [("xyz",  0.00016, 1e-5, 5e-4),  # 100
            ("scaling", 0.005,  1e-4, 1e-2),  # 100
            ("rotation", 0.001, 1e-4, 5e-3), # 50
            ("opacity", 0.025, 1e-3, 5e-2), # 50
            ("f_dc", 0.0025, 1e-4, 5e-3),]  # 50

densi_hp_space = [("densify_grad_threshold", 0.0002, 0.00018, 0.00022) , ("cameras_extent_scale",1.0,0.9, 1.1)]
score_coefficients = {'view_importance': 50, 'edge_importance': 50, 'mse_importance': 50, 'grad_importance': 25, 'dist_importance': 50, 'opac_importance': 100, 'dept_importance': 5, 'loss_importance': 10, 'radii_importance': 10, 'scale_importance': 25, 'count_importance': 0.1, 'blend_importance': 50}

#densi_hp_space = [("densify_grad_threshold", 0.0002, 0.00014, 0.00026) , ("cameras_extent_scale",1.0,0.7, 1.3)]
from utils.taming_utils import compute_gaussian_score, get_edges, get_count_array, get_count_arrayv2

# render": rendered_image,
#             "viewspace_points": screenspace_points,
#             "visibility_filter" : (radii > 0).nonzero(),
#             "radii": radii,
#             "counts" : counts,
#             "lists" : lists,
#             "listsRender": listsRender,
#             "listsDistance": listsDistance,
#             "gaussian_centers": centers,
#             "gaussian_depths": depths,
#             "gaussian_radii": my_radii,
#             "accum_weights": accum_weights,
#             "accum_count" : accum_count,
#             "accum_blend" : accum_blend,
#             "accum_dist" :  accum_dist
            
            
def render_loss(viewpoint_cam, gaussians, pipe, bg, dataset,gt_image):
   
    render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
    
    image = render_pkg["render"]
    viewspace_point_tensor = render_pkg["viewspace_points"]
    visibility_filter = render_pkg["visibility_filter"]
    radii = render_pkg["radii"]
    
    Ll1 = l1_loss(image, gt_image)
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    else:
        ssim_value = ssim(image, gt_image)
    
    attr_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
    return attr_loss, visibility_filter

def render_loss_densi(viewpoint_cam, gaussians, pipe, bg, dataset,gt_image):
    
    render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            
    image = render_pkg["render"]
    viewspace_point_tensor = render_pkg["viewspace_points"]
    visibility_filter = render_pkg["visibility_filter"]
    radii = render_pkg["radii"]
    
    Ll1 = l1_loss(image, gt_image)
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    else:
        ssim_value = ssim(image, gt_image)
    
    attr_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
    return attr_loss, visibility_filter, viewspace_point_tensor, radii


def negative_psnr_loss(viewpoint_cam, gaussians, pipe, bg, dataset, gt_image):
    
    render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            
    image = render_pkg["render"]
    viewspace_point_tensor = render_pkg["viewspace_points"]
    visibility_filter = render_pkg["visibility_filter"]
    radii = render_pkg["radii"]
    
    psnr_value = - psnr(image, gt_image).mean()
    
    return psnr_value, visibility_filter





def kl_gauss(p_mu, p_var, q_mu, q_var):
    # KL(N(p_mu,p_var) || N(q_mu,q_var))
    # = 0.5 * [ log(q_var/p_var) + (p_var + (p_mu−q_mu)^2)/q_var − 1 ]
    return 0.5 * (
        torch.log(q_var / p_var)
        + (p_var + (p_mu - q_mu).pow(2)) / q_var
        - 1
    )

# def psnr_reward_with_gauss_kl(
#     viewpoint_cam,
#     gaussians,
#     pipe,
#     bg,
#     dataset,
#     gt_image,
#     lambda_penalty=5.0
# ):

#     render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
#     image             = render_pkg["render"]          # [B, C, H, W]
#     visibility_filter = render_pkg["visibility_filter"]

#     psnr_val = psnr(image, gt_image).mean()
    
#     # PSNR


#     flat = image.view(-1)
#     p_mu  = flat.mean()
#     p_var = flat.var(unbiased=False) + 1e-6


#     #q_mu, q_var = 0.5, 1.0/12.0
    
    
#     q_mu = dataset[0][viewpoint_cam.image_name] 
#     q_var = dataset[1][viewpoint_cam.image_name] 
        

#     kl = kl_gauss(p_mu, p_var, q_mu, q_var)

#     reward = psnr_val - lambda_penalty * kl
#     return reward, visibility_filter
def psnr_reward_with_kl(
    viewpoint_cam,
    gaussians,
    pipe,
    bg,
    dataset,
    gt_image,
    lambda_kl=1.0,
    num_bins=256
):

    render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
    image             = render_pkg["render"]        # [B, C, H, W]
    visibility_filter = render_pkg["visibility_filter"]

    # 2) PSNR
    psnr_value = psnr(image, gt_image).mean()



    flat = image.detach().cpu().view(-1)
    hist = torch.histc(flat, bins=num_bins, min=0.0, max=1.0)
    p = hist / hist.sum()


    q = torch.ones_like(p) / num_bins



    kl = F.kl_div(p.log(), q, reduction="sum")


    reward = psnr_value - lambda_kl * kl

    return reward, visibility_filter
# absolute violation penalty
def psnr_reward_relu(
    viewpoint_cam,
    gaussians,
    pipe,
    bg,
    dataset,
    gt_image,
    lambda_penalty=1.0
):

    render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
    image             = render_pkg["render"]          # [B, C, H, W]
    visibility_filter = render_pkg["visibility_filter"]

    psnr_value = psnr(image, gt_image).mean()
    

    over_low  = F.relu(-image)
    over_high = F.relu(image - 1.0)
    penalty   = (over_low + over_high).mean()
    

    reward = psnr_value -  lambda_penalty * penalty
    
    return reward, visibility_filter

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False



def compute_policy_loss_entropy(lr_adjust_dict, reward, entropy_coef=0.01):
    """
    lr_adjust_dict 每个 entry 包含:
      - "log_prob": Tensor([B])
      - "entropy" : Tensor([B])
    reward: Tensor([B]) or scalar Tensor
    """
    policy_loss  = 0.0
    entropy_loss = 0.0

    for v in lr_adjust_dict.values():
        policy_loss  += (-v["log_prob"] * reward).mean()
        entropy_loss +=  v["entropy"].mean()

    return policy_loss - entropy_coef * entropy_loss

def get_log_and_entropy(lr_adjust_dict):
    log_prob = 0.0
    entro_py = 0.0 
    for v in lr_adjust_dict.values():
        log_prob += v["log_prob"].mean()
        entro_py += v["entropy"].mean()
    return log_prob, entro_py
def traverse_backwards(fn, visited=set(), depth=0):
    if fn in visited or fn is None:
        return
    visited.add(fn)
    print("  " * depth + str(fn))
    for u in fn.next_functions:
        if u[0] is not None:
            traverse_backwards(u[0], visited, depth + 1)

def adjust_densification_parameter(opt, densi_adjust_dict, densi_parameter_ori_dict):
    for k in densi_adjust_dict:
        setattr(opt, k, densi_adjust_dict[k]['action'].item() * densi_parameter_ori_dict[k])

def restore_densification_parameter(opt, densi_parameter_ori_dict):
    for k in densi_parameter_ori_dict:
        setattr(opt, k, densi_parameter_ori_dict[k])

def get_multi_step_weight(lr_init, lr_final, mile_step):
    def helper(step):
        if step < mile_step:
            return lr_init
        else:
            return lr_final

    return helper

def get_penalty_function(opt):

    def psnr_reward_with_gauss_kl(
        viewpoint_cam,
        gaussians,
        pipe,
        bg,
        dataset,
        gt_image
    ):
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image             = render_pkg["render"]
        visibility_filter = render_pkg["visibility_filter"]

        psnr_val = psnr(image, gt_image).mean()
        flat = image.view(-1)
        p_mu  = flat.mean()
        p_var = flat.var(unbiased=False) + 1e-6

        q_mu  = dataset[0][viewpoint_cam.image_name]
        q_var = dataset[1][viewpoint_cam.image_name]
        kl    = kl_gauss(p_mu, p_var, q_mu, q_var)

        reward = psnr_val - opt.lambda_penalty * kl
        return reward, visibility_filter

    return psnr_reward_with_gauss_kl

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    
    def psnr_reward(viewpoint_cam, gaussians, pipe, bg, dataset, gt_image):
    
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
                
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        
        psnr_value = psnr(image, gt_image).mean()
        
        penalty = (image - image.clamp(0, 1)).abs().mean()

        reward = psnr_value -  LAMBDA_PENALTY * penalty
        
        return reward, visibility_filter


    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    lr_policy_net = None 
    densi_policy_net = None  
    gaussians = SGaussianModelMem2(dataset.sh_degree, opt, lr_policy_net, densi_policy_net )
    
    
    base_reward_function = psnr_reward
    reward_function = psnr_reward
    
    
    scene = Scene(dataset, gaussians, resolution_scales=[1.0,2.0])
    scene.offset = opt.offset
    
     
    num_views = len(scene.getTrainCameras().copy())
    
    if opt.policy ==1:
        from helper_model import ScalarPolicyWithSchedStateGRUKeywise
        gaussians.lr_policy_network = ScalarPolicyWithSchedStateGRUKeywise( key_info_list=lr_search_space, hidden=opt.hidden_state,  init_std=opt.init_std, clamp=True,detach=True, max_steps=opt.iterations,ema_scale=opt.ema)
        gaussians.densi_policy_network = ScalarPolicyWithSchedStateGRUKeywise( key_info_list=densi_hp_space, hidden=opt.hidden_state, init_std=opt.densi_init_std, clamp=True, detach=True, max_steps=opt.iterations)
  
    elif opt.policy ==2:
        from helper_modelv2 import ScalarPolicyWithSchedStateGRUKeywiseCrossKey
        gaussians.lr_policy_network = ScalarPolicyWithSchedStateGRUKeywiseCrossKey( key_info_list=lr_search_space,  init_std=opt.init_std, clamp=True,detach=True, max_steps=opt.iterations)
        gaussians.densi_policy_network = ScalarPolicyWithSchedStateGRUKeywiseCrossKey( key_info_list=densi_hp_space,  init_std=opt.densi_init_std, clamp=True, detach=True, max_steps=opt.iterations)
    
    # ScalarPolicyWithSchedStateLSTMKeywise
    elif opt.policy == 3:
        from helper_modelv2 import ScalarPolicyWithSchedStateLSTMKeywise
        gaussians.lr_policy_network = ScalarPolicyWithSchedStateLSTMKeywise( key_info_list=lr_search_space,  init_std=opt.init_std, clamp=True,detach=True, max_steps=opt.iterations)
        gaussians.densi_policy_network = ScalarPolicyWithSchedStateLSTMKeywise( key_info_list=densi_hp_space,  init_std=opt.densi_init_std, clamp=True, detach=True, max_steps=opt.iterations)
    
    elif opt.policy == 4:
        # MixtureGRUPolicy
        
        from helper_model import MixtureGRUPolicy
        gaussians.lr_policy_network = MixtureGRUPolicy( key_info_list=lr_search_space, hidden=opt.hidden_state,  init_std=opt.init_std, clamp=True,detach=True, max_steps=opt.iterations)
        gaussians.densi_policy_network = MixtureGRUPolicy( key_info_list=densi_hp_space, hidden=opt.hidden_state, init_std=opt.densi_init_std, clamp=True, detach=True, max_steps=opt.iterations)
  
    
    gaussians.training_rl_setup(opt, attribute_list=ATTRIBUTES1,attribute_list_rest=ATTRIBUTES2)   
    gaussians.training_setup(opt) 
    viewpoint_dict = {}
    training_views = scene.getTrainCameras().copy()
    for i in range(len(training_views)):
        viewpoint_dict[training_views[i].image_name] = i
    
    all_edges = []
    for view in scene.getTrainCameras():
        edges_loss = get_edges(view.original_image).squeeze().cuda()
        edges_loss_norm = (edges_loss - torch.min(edges_loss)) / (torch.max(edges_loss) - torch.min(edges_loss))
        all_edges.append(edges_loss_norm.cpu())
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    #depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    
    

    gaussians.policy_lr_scheduler = get_expon_lr_func(opt.pl_lr,  opt.pl_lr_final, max_steps=opt.iterations)
    gaussians.policy_densi_scheduler = get_expon_lr_func(opt.pl_lr,  opt.pl_lr_final, max_steps=opt.iterations)
    
    # beta final 0.03
    beta_weight = get_expon_lr_func(lr_init=0.01,  lr_final=opt.beta_final, max_steps=opt.iterations/opt.beta_itr)
 

    viewpoint_stack = scene.getTrainCamerasV2().copy()
    
    viewpoint_stack_half = scene.getTrainCamerasV2(scale=2.0).copy()
    for idx in range(len(viewpoint_stack)):
        assert viewpoint_stack[idx].image_name == viewpoint_stack_half[idx].image_name
    
    viewpoint_indices = list(range(len(viewpoint_stack)))
    
    
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    good_iters = 0
    

    
    iteration = first_iter
    
    # set up default Hyper Parameters
    lr_parameter_ori_dict = {}
    for idx, param_group in enumerate(gaussians.optimizer.param_groups):
        lr_parameter_ori_dict[param_group['name']] = param_group['lr']  
    densi_parameter_ori_dict = {}
    for k in ["densify_grad_threshold", "cameras_extent_scale"]:
        densi_parameter_ori_dict[k] = getattr(opt, k)
   
    rl_densi_cnt = 0

    pre_loss = torch.tensor(0.0).cuda()
    training_list = []
    #for iteration in range(first_iter, opt.iterations + 1):
    iteration = 0
    boost_step = 0
    need_release_opacity = False 
    counts_array = None
    densify_iter_num = 0
    training_list_half = []
    def getq_mu(image):
        flat = image.reshape(-1)
        p_mu  = flat.mean()
        p_var = flat.var(unbiased=False) + 1e-6
        return p_mu, p_var
    dataset_mu = {}
    dataset_var = {}
    

    if opt.reward_penalty > 1:
        all_viewpoint = scene.getTrainCameras().copy()
        for reward_viewpoint_cam in all_viewpoint:
            # Transfer the original image to the CUDA device.
            gt_image = reward_viewpoint_cam.original_image.cuda()
            q, mu = getq_mu(gt_image)
            dataset_mu[reward_viewpoint_cam.image_name] = q
            dataset_var[reward_viewpoint_cam.image_name] = mu

        dataset_stat= [dataset_mu, dataset_var]
    else:
        dataset_stat = [None, None ]
    
    
    #    gaussians.lr_policy_network
    #gaussians.densi_policy_network
    
    if opt.policy !=3:
    
        hidden_lr = [torch.zeros(1, gaussians.lr_policy_network.hidden_size, device="cuda") for k in range(gaussians.lr_policy_network.num_keys)]

        hidden_ds = [torch.zeros(1, gaussians.densi_policy_network.hidden_size, device="cuda") for k in range(gaussians.lr_policy_network.num_keys)]
        
        gaussians.lr_policy_network.hidden = hidden_lr
        gaussians.densi_policy_network.hidden = hidden_ds
    else:
        
        pass 
        
    
        # record time
    time_taken = {
            "dataset": 0.0,
            "RLLR-nn": 0.0,
            "RLLR-simu": 0.0,
            "RLLR-other": 0.0,
            "RLDS-nn": 0.0,
            "RLDS-ds": 0.0,
            "RLDS-simu": 0.0,
            "RLDS-other": 0.0,
            "total":0.0
    }
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    job_start_time = time.time()
    
    while iteration < (opt.iterations  +  opt.future_step+1):

        is_lr_rl_step = True 
            
        if iteration < opt.boost_step:
            boost_step = False 
        else:
            if iteration > opt.boost_step and iteration < opt.boost_step + opt.boost_stepoffset :
                boost_step = True 
            else:
                boost_step = False 
            

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)
        if counts_array == None:
            counts_array = get_count_array(len(scene.gaussians.get_xyz), args.budget, opt, mode=args.mode)
            print(counts_array)
            
        
        
        #if iteration < opt.nv_iters:
        policy_lr = gaussians.update_learning_rate_policy(iteration)
        
        

        
            
        lr_parameter_ori_dict['xyz'] = xyz_lr # as the

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0 and iteration !=0 :
            gaussians.oneupSHdegree()

        

    

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        if TIMING:
            start_time.record()
        
        while len(training_list) < opt.future_step:
            
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCamerasV2().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
                
                viewpoint_stack_half = scene.getTrainCamerasV2(scale=2.0)
                
                
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            vind = viewpoint_indices.pop(rand_idx)
            viewpoint_cam_half = viewpoint_stack_half.pop(rand_idx)
            training_list.append(viewpoint_cam)
            training_list_half.append(viewpoint_cam_half)
            
        if TIMING:
            end_time.record()
            torch.cuda.synchronize()

            time_taken["dataset"] += start_time.elapsed_time(end_time)/1000
        
        
        if is_lr_rl_step :
            local_step = 0 
            reward_cameras_list =  scene.getRewardCaemeras(scale=1.0).copy()
            
            if boost_step:
                inner_loop = opt.boost_step_loop 
            else:
                inner_loop = opt.inner_loop 
            
            state_before_densi = gaussians.snapshot_state(False)
            
            
            
     
                        
           
            best_reward = -1000
            best_policy = None 
            
            while local_step < inner_loop:
                
                
                
                if TIMING:
                    start_time.record()
                lr_adjust_dict = gaussians.lr_policy_network(loss=pre_loss, step=iteration )
                
                if TIMING:
                    end_time.record()
                    torch.cuda.synchronize()
                    time_taken["RLLR-nn"] += start_time.elapsed_time(end_time)/1000
                    
                if TIMING:
                    start_time.record()
                gaussians.load_snapshot_state(state_before_densi) #TODO two stream optimize
                
                
                
                gaussians.adjust_lr_by_actions(lr_adjust_dict=lr_adjust_dict, parameter_name_dict=lr_parameter_ori_dict)
                
                
                origin_psnr = evaluate_total_gain_psnr(reward_cameras_list, gaussians, pipe, bg, dataset_stat, base_reward_function)
                
                if TIMING:
                    start_time.record()
                try_gradient_updating(training_list_half, gaussians, pipe, bg, dataset, render_loss)
                if TIMING:
                    end_time.record()
                    torch.cuda.synchronize()
                    time_taken["RLLR-simu"] += start_time.elapsed_time(end_time) /1000
                    
                if TIMING:
                    start_time.record()
                rl_psnr = evaluate_total_gain_psnr(reward_cameras_list, gaussians, pipe, bg, dataset_stat, reward_function)
            
                length_reward = len(reward_cameras_list)    
                reward = opt.rw_scaler  * (rl_psnr - origin_psnr) / length_reward  # default 10.0
                
                if reward > best_reward:
                    best_reward = reward 
                    best_policy = lr_adjust_dict
                    best_rl_psnr = rl_psnr
                        
                    best_model = gaussians.snapshot_state(False)
                
                beta = beta_weight(iteration)
                policy_loss = compute_policy_loss_entropy(lr_adjust_dict, reward.detach(),entropy_coef=beta) #  
                gaussians.lr_policy_optimizer.zero_grad()
                policy_loss.backward()
                
                  
                torch.nn.utils.clip_grad_norm_(gaussians.lr_policy_network.parameters(), max_norm=opt.grdclip)
                gaussians.lr_policy_optimizer.step()
                local_step+=1 
                
                if TIMING:
                    end_time.record()
                    torch.cuda.synchronize()
                    time_taken["RLLR-other"] += start_time.elapsed_time(end_time)/1000
                    
                
            
            last_psnr = rl_psnr
            gaussians.load_snapshot_state(state_before_densi)
            
            
                  
    
    
        
        xyz_lr = gaussians.update_learning_rate(iteration)
        lr_parameter_ori_dict['xyz'] = xyz_lr
        
        # if boost_step:
        #     if best_policy is None:
        #         best_policy = lr_adjust_dict
        gaussians.adjust_lr_by_actions(lr_adjust_dict=best_policy, parameter_name_dict=lr_parameter_ori_dict)

        rest_train_step = opt.future_step
        
        while rest_train_step > 0:
        
            
            rest_train_step -= 1 
            viewpoint_cam = training_list.pop(0)
            training_list_half.pop(0)
            
            
        
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            
            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # Depth regularization
            Ll1depth_pure = 0.0
            
            Ll1depth = 0
            
            
            
            gaussians.optimizer.zero_grad()
            loss.backward()
            #grads = 
            
            
            gaussians.optimizer.step()
            
            if args.sh_lower:
                if iteration % 16 == 0:
                    gaussians.shoptimizer.step()
                    gaussians.shoptimizer.zero_grad(set_to_none = True)
            else:
                gaussians.shoptimizer.step()
                gaussians.shoptimizer.zero_grad(set_to_none = True)
            
            iter_end.record()

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), dataset.train_test_exp,good_iters=good_iters)


                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))


                    scene.save(iteration)



            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                
                with torch.no_grad():
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    
                    if TIMING:
                        start_time.record()
            
                    gaussians.restore_default_lr(lr_parameter_ori_dict)
                    
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                
                    if rl_densi_cnt < opt.rl_densi_cnt and iteration % (opt.densification_interval) == 0:
                        
                        # sepcial hack to make the same future cameras as updated in the training roop
                        while len(training_list) < opt.future_step:
                            
                            
                            
                            if not viewpoint_stack:
                                viewpoint_stack = scene.getTrainCamerasV2().copy()
                                viewpoint_indices = list(range(len(viewpoint_stack)))
                                
                                viewpoint_stack_half = scene.getTrainCamerasV2(scale=2.0)
                                
                                
                            rand_idx = randint(0, len(viewpoint_indices) - 1)
                            viewpoint_cam = viewpoint_stack.pop(rand_idx)
                            vind = viewpoint_indices.pop(rand_idx)
                            viewpoint_cam_half = viewpoint_stack_half.pop(rand_idx)
                            training_list.append(viewpoint_cam)
                            training_list_half.append(viewpoint_cam_half)
            
                        
                        reward_cameras_list_densi =  scene.getRewardCaemeras(scale=1.0).copy()
                        #scene.getTestCameras()
                        #scene.getRewardCaemeras().copy()
                        den_si_training_list = None 
                        
                        length_reward = len(reward_cameras_list_densi)
                        
                        
                        
                        best_densi_reward = -1000
                        densi_inner_loop = opt.densi_rl_innter_step 
    
                        #state_before_densi = [gaussians.snapshot_state(False) for i in range(densi_inner_loop+1)]
                        #state_before_densi = gaussians.snapshot_state(False)
                        state_before_densi = best_model
                        
                        restore_densification_parameter(opt, densi_parameter_ori_dict)
                        
                        with torch.no_grad():
                            densify_iter_num = densify_iteration(
                                scene=scene,
                                all_edges=all_edges,
                                cams=args.cams,
                                gaussians=gaussians,
                                pipe=pipe,
                                bg=bg,
                                score_coefficients=score_coefficients,
                                opt=opt,
                                size_threshold=size_threshold,
                                counts_array=counts_array,
                                radii=radii,
                                iter_num=densify_iter_num)
                        
                        
                        default_pack_ori = try_gradient_updating_densi(training_list_half, gaussians, pipe, bg, dataset, render_loss_densi)
                        default_densi_state = gaussians.snapshot_state(include_grads=False)
                        default_psnr = evaluate_total_gain_psnr(reward_cameras_list_densi, gaussians, pipe, bg, dataset_stat, base_reward_function) 
                
                        local_step = 0 

                        total_densi_loss = 0
                        while local_step < densi_inner_loop:
                            
                            local_step +=1
                            
                            gaussians.load_snapshot_state_densi(state_before_densi)
                            if TIMING:
                                start_time.record()
                            densi_adjust_dict = gaussians.densi_policy_network(loss=pre_loss, step=iteration)
                            
                            if TIMING:
                                end_time.record()
                                torch.cuda.synchronize()

                                time_taken["RLDS-nn"] += start_time.elapsed_time(end_time)/1000
            

                            adjust_densification_parameter(opt, densi_adjust_dict, densi_parameter_ori_dict)
                            densify_iter_num = densify_iter_num-1
                            
                            if TIMING:
                                start_time.record()
                            with torch.no_grad():
                                densify_iter_num = densify_iteration(
                                    scene=scene,
                                    all_edges=all_edges,
                                    cams=args.cams,
                                    gaussians=gaussians,
                                    pipe=pipe,
                                    bg=bg,
                                    score_coefficients=score_coefficients,
                                    opt=opt,
                                    size_threshold=size_threshold,
                                    counts_array=counts_array,
                                    radii=radii,
                                    iter_num=densify_iter_num,
                                )
                                    
                            if TIMING:
                                end_time.record()
                                torch.cuda.synchronize()

                                time_taken["RLDS-ds"] += start_time.elapsed_time(end_time)/1000
                            
                            tmp_state = gaussians.snapshot_state(False)
                            if TIMING:
                                start_time.record()
                            default_pack_ori = try_gradient_updating_densi(training_list_half, gaussians, pipe, bg, dataset, render_loss_densi)
                            
                            if TIMING:
                                end_time.record()
                                torch.cuda.synchronize()

                                time_taken["RLDS-simu"] += start_time.elapsed_time(end_time)/1000
                            
                            if TIMING:
                                start_time.record()
                            densi_psnr = evaluate_total_gain_psnr(reward_cameras_list_densi, gaussians, pipe, bg, dataset_stat, reward_function) 
                            
                            densi_reward = opt.rw_scaler * (densi_psnr - default_psnr)  / length_reward #(densi_psnr  ) / length_reward * 30.0 # (densi_psnr  ) / length_reward * 30.0
                            densi_policy_loss = compute_policy_loss_entropy(densi_adjust_dict, densi_reward.detach())
                            gaussians.densi_policy_optimizer.zero_grad(set_to_none=True)
                            densi_policy_loss.backward()
                            torch.nn.utils.clip_grad_norm_(gaussians.densi_policy_network.parameters(), max_norm=opt.grdclip)    
                            gaussians.densi_policy_optimizer.step()
                            if TIMING:
                                end_time.record()
                                torch.cuda.synchronize()

                                time_taken["RLDS-other"] += start_time.elapsed_time(end_time) /1000
                            
                            if densi_reward > best_densi_reward:
                                best_densi_reward = densi_reward
                                best_densi_state = tmp_state
                                
                        # ---------- 1. baseline densify ----------
                        gaussians.load_snapshot_state_densi(best_densi_state) 

                        
                        rl_densi_cnt += 1
                        
                        
                        
            
                    else:
                        quit()
                        
            
            
             
            if iteration % opt.shuffle_interval == 0:                
                #scene.shuffle()
                idx = list(range(len(scene.train_cameras[1.0])))
                random.shuffle(idx)
                scene.train_cameras[1.0] = [scene.train_cameras[1.0][i] for i in idx]
                
                if 2.0 in scene.train_cameras:
                    scene.train_cameras[2.0] = [scene.train_cameras[2.0][i] for i in idx]
                
            if iteration % opt.hidden_interval == 0:
                
                hidden_lr = [torch.zeros(1, gaussians.lr_policy_network.hidden_size, device="cuda") for k in range(gaussians.lr_policy_network.num_keys)]

                hidden_ds = [torch.zeros(1, gaussians.densi_policy_network.hidden_size, device="cuda") for k in range(gaussians.lr_policy_network.num_keys)]
                
                gaussians.lr_policy_network.hidden = hidden_lr
                if iteration > 1100:
                    gaussians.densi_policy_network.hidden = hidden_ds
             
            if iteration == args.ho_iteration:
                print("Release opacity limit")
                need_release_opacity=True
                
                
                hidden_lr = [torch.zeros(1, gaussians.lr_policy_network.hidden_size, device="cuda") for k in range(gaussians.lr_policy_network.num_keys)]
                hidden_ds = [torch.zeros(1, gaussians.densi_policy_network.hidden_size, device="cuda") for k in range(gaussians.lr_policy_network.num_keys)]
                gaussians.lr_policy_network.hidden = hidden_lr
                gaussians.densi_policy_network.hidden = hidden_ds
                

        
            
            pre_loss = loss.detach()
            iteration +=1

 
        old_iteration = iteration - opt.future_step
                
        
        
        if old_iteration < opt.densify_until_iter:
            if old_iteration % opt.opacity_reset_interval == 0 :
                if old_iteration ==0:
                    print("wrong")
                    #quit()
                else:
                    gaussians.reset_opacity()
                    
                    
        if need_release_opacity :
            with torch.no_grad():
                gaussians.modify_functions()
                need_release_opacity=False
                

        # if (old_iteration in saving_iterations):
        #     scene.save(old_iteration)
            

        if iteration % 100:
            torch.cuda.empty_cache()
        
    if TIMING:   
        timing_json_path = os.path.join(args.model_path, "timing.json")
        time_taken["total"] = time.time()-job_start_time
        # time_results = {}
        # for k in time_taken:
        #     time_results[k] = np.mean(np.array(time_taken[k]))
        
        peak = torch.cuda.max_memory_allocated("cuda") / 1024**2
        time_taken["PeakMem"] = peak
        with open(timing_json_path, 'w', encoding='utf-8') as fp:
            json.dump(time_taken, fp, ensure_ascii=False, indent=4)
            
        with open(timing_json_path, 'w', encoding='utf-8') as fp:
            json.dump(time_taken, fp, ensure_ascii=False, indent=4)
        

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp,good_iters):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations : 
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                log_entry = (f"{config['name']}-psnr_{psnr_test}-iteration_{iteration}\n")
                with open(os.path.join(scene.model_path , "result.txt"), "a") as file:
                    file.write(log_entry)
                
                good_rate = good_iters / (iteration - 100) if iteration != 100 else 0
                rate_entry = (f"{config['name']}-rate_{good_rate}-iteration_{iteration}\n")
                with open(os.path.join(scene.model_path , "log.txt"), "a") as file:
                    file.write(rate_entry)
                
                # free memory 
                for idx, viewpoint in enumerate(config['cameras']):
                    viewpoint.original_image.detach()  
                    torch.cuda.empty_cache()
                    

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            
            
        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    # parser.add_argument('--debug_from', type=int, default=-1)
    # parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[100, 1000, 2000,3_000, 7_000, 10_000, 14_000, 16_000,18_000, 20_000, 22000, 25000,27000,29000,29500, 30_000])
    # #parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])

    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[]) # [7_000, 30_000]
    # parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    # parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    # parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000]) # remove for saving time
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--cams", type=int, default=10)
    parser.add_argument("--budget", type=float, default=20)
    parser.add_argument("--mode", type=str, default="multiplier", choices=["multiplier", "final_count"])
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--ho_iteration", type=int, default=15000)
    parser.add_argument("--sh_lower", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
   
   
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    opt = op.extract(args)
    
    defaults_opt = {action.dest: action.default for action in parser._actions if action.dest != argparse.SUPPRESS}
    from helper_data import generate_model_name
    sub_name, changed_dict = generate_model_name(opt, defaults_opt, "")
    #args.model_path = os.path.join(args.model_path, sub_name)
    json_path = os.path.join(args.model_path, "config.json")
    import json 
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(changed_dict, fp, ensure_ascii=False, indent=4)
    # Initialize system state (RNG)
    os.environ["WORLD_SCALE"] = str(opt.world_scale)
    os.environ["lambda_penalty"] = str(opt.lambda_penalty)
    safe_state(args.quiet, int(opt.seeds)) # added exp for seed in rebuttal

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    LAMBDA_PENALTY = float(os.environ.get('lambda_penalty', '0.0'))
    TIMING = True if opt.timing == 1 else False 


    training(lp.extract(args), opt, pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
