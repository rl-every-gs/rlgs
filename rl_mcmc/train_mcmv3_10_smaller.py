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
import json
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.gaussian_model import build_scaling_rotation
from scene.gaussian_model_rltmgs import SGaussianModelMem2
from helper_common import ATTRIBUTES1, ATTRIBUTES2
from rl_policy import densi_hp_space, psnr_reward, compute_policy_loss_entropy
from utils.general_utils import safe_state, get_expon_lr_func
from helper_rl import try_gradient_updating, evaluate_total_gain_psnr, change_seed, try_gradient_updating_densi, get_psnr_dict, evaluate_total_gain_psnr_rt_list
import argparse
try:
    from tensorboardX import SummaryWriter as SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False
            
def render_loss(viewpoint_cam, gaussians, pipe, bg, args, gt_image):
   
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
    
    attr_loss = attr_loss + gaussians.os * args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
    attr_loss = attr_loss + gaussians.sr * args.scale_reg * torch.abs(gaussians.get_scaling).mean()
            
    return attr_loss, visibility_filter


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    lr_policy_net = None 
    densi_policy_net = None  
    gaussians = SGaussianModelMem2(dataset.sh_degree, opt, lr_policy_net, densi_policy_net )

    base_reward_function = psnr_reward
    reward_function = psnr_reward
    scene = Scene(dataset, gaussians, seed=opt.seeds)
    scene.offset = opt.offset
    
    lr_search_space= [("xyz",  0.00016, 1e-5, 5e-4),  # 100
            ("scaling", 0.005,  1e-4, 1e-2),  # 100
            ("rotation", 0.001, 1e-4, 5e-3), # 50
            ("opacity", 0.025, 1e-3, 5e-2), # 50
            ("f_dc", 0.0025, 1e-4, 5e-3),]  # 50

    lr_search_space=lr_search_space+[("os",  1.0, 0.05, 5.0),("sr",1.0, 0.05, 5.0)]
    
    if opt.policy ==1:
        from helper_model import ScalarPolicyWithSchedStateGRUKeywise
        gaussians.lr_policy_network = ScalarPolicyWithSchedStateGRUKeywise( key_info_list=lr_search_space, hidden=opt.hidden_state,  init_std=opt.init_std, clamp=True,detach=True, max_steps=opt.iterations)
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
        
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 
    gaussians.policy_lr_scheduler = get_expon_lr_func(opt.pl_lr,  opt.pl_lr_final, max_steps=opt.iterations)
    gaussians.policy_densi_scheduler = get_expon_lr_func(opt.pl_lr,  opt.pl_lr_final, max_steps=opt.iterations)
    
    gaussians.policy_lr_scheduler = get_expon_lr_func(opt.pl_lr,  opt.pl_lr_final, max_steps=opt.iterations)
    gaussians.policy_densi_scheduler = get_expon_lr_func(opt.pl_lr,  opt.pl_lr_final, max_steps=opt.iterations)
    
    # beta final 0.03
    beta_weight = get_expon_lr_func(lr_init=0.01,  lr_final=opt.beta_final, max_steps=opt.iterations/opt.beta_itr)
 

    viewpoint_stack = scene.getTrainCamerasV2().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    #

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
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
    
    
    #gaussians.lr_policy_network
    #gaussians.densi_policy_network
    
    if opt.policy !=3:
    
        hidden_lr = [torch.zeros(1, gaussians.lr_policy_network.hidden_size, device="cuda") for k in range(gaussians.lr_policy_network.num_keys)]

        hidden_ds = [torch.zeros(1, gaussians.densi_policy_network.hidden_size, device="cuda") for k in range(gaussians.lr_policy_network.num_keys)]
        
        gaussians.lr_policy_network.hidden = hidden_lr
        gaussians.densi_policy_network.hidden = hidden_ds
    else:
        
        pass 
    
    iteration = 0
    

            
    while iteration < (opt.iterations  +  opt.future_step+1):
        
        is_lr_rl_step = True 
            
        if iteration < opt.boost_step:
            boost_step = False 
        else:
            boost_step = True 
            
            

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)
        policy_lr = gaussians.update_learning_rate_policy(iteration)
        
        if iteration %20 ==0:
            tb_writer.add_scalar('policy/lr', policy_lr, iteration)
            
            
        lr_parameter_ori_dict['xyz'] = xyz_lr # as the

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0 and iteration !=0 :
            gaussians.oneupSHdegree()


        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        
        
        while len(training_list) < opt.future_step:
            
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCamerasV2().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            vind = viewpoint_indices.pop(rand_idx)
            training_list.append(viewpoint_cam)
            
            
        if is_lr_rl_step :
            local_step = 0 
            reward_cameras_list =  scene.getRewardCaemeras().copy()
            
            if boost_step:
                inner_loop = opt.boost_step_loop-1 #8
            else:
                inner_loop = 1 
            
            state_before_densi = gaussians.snapshot_state(False)
            
            train_camera_list =  training_list 
           
            best_reward = -1000
            best_policy = None 
            
            while local_step < inner_loop:
                lr_adjust_dict = gaussians.lr_policy_network(loss=pre_loss, step=iteration )
                
                gaussians.load_snapshot_state(state_before_densi)
                
                gaussians.adjust_lr_by_actions_v2(lr_adjust_dict=lr_adjust_dict, parameter_name_dict=lr_parameter_ori_dict)
                origin_psnr = evaluate_total_gain_psnr(reward_cameras_list, gaussians, pipe, bg, dataset_stat, base_reward_function)
                
                gaussians.check_num_g_consistent(local_step)
                
                # dataset place holder with args
                try_gradient_updating(train_camera_list, gaussians, pipe, bg, args, render_loss)
            
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
                
                if opt.clip_norm == 0:
                    pass 
                else:   
                    torch.nn.utils.clip_grad_norm_(gaussians.lr_policy_network.parameters(), max_norm=opt.grdclip)
                gaussians.lr_policy_optimizer.step()
                local_step+=1 
                
                if local_step == inner_loop and iteration % 20 ==0 :
                    tb_writer.add_scalar(f'rl/last_reward', reward.item(), global_step=iteration)
                    
                    tb_writer.add_scalar(f'learning_rate/LR', float(lr_adjust_dict["opacity"]['action']), global_step=iteration)
        
            last_psnr = rl_psnr
            gaussians.load_snapshot_state(state_before_densi)
            
            
        
        xyz_lr = gaussians.update_learning_rate(iteration)
        lr_parameter_ori_dict['xyz'] = xyz_lr
        

        gaussians.adjust_lr_by_actions_v2(lr_adjust_dict=best_policy, parameter_name_dict=lr_parameter_ori_dict)

        rest_train_step = opt.future_step
        
        while rest_train_step > 0:
            rest_train_step -= 1 
            viewpoint_cam = training_list.pop(0)
            
        
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            
            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            
            ssim_value = ssim(image, gt_image)

            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            loss = loss + gaussians.os*args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
            loss = loss + gaussians.sr*args.scale_reg * torch.abs(gaussians.get_scaling).mean()

            loss.backward()

            iter_end.record()

            

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                    
                if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                    gaussians.relocate_gs(dead_mask=dead_mask)
                    gaussians.add_new_gs(cap_max=args.cap_max)

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                    L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                    actual_covariance = L @ L.transpose(1, 2)

                    def op_sigmoid(x, k=100, x0=0.995):
                        return 1 / (1 + torch.exp(-k * (x - x0)))
                    
                    noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*args.noise_lr*xyz_lr
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                    gaussians._xyz.add_(noise)
                
                if iteration % opt.shuffle_interval == 0:                
                    scene.shuffle()
                    hidden_lr = [torch.zeros(1, gaussians.lr_policy_network.hidden_size, device="cuda") for k in range(gaussians.lr_policy_network.num_keys)]

                    hidden_ds = [torch.zeros(1, gaussians.densi_policy_network.hidden_size, device="cuda") for k in range(gaussians.lr_policy_network.num_keys)]
                        
                    gaussians.lr_policy_network.hidden = hidden_lr
                    gaussians.densi_policy_network.hidden = hidden_ds
                        
                pre_loss = loss.detach()
                iteration +=1
            
        old_iteration = iteration- opt.future_step

        current_psnr  = evaluate_total_gain_psnr(reward_cameras_list, gaussians, pipe, bg, dataset_stat, reward_function)
        if current_psnr < best_rl_psnr:
            
            # and old_iteration % opt.densification_interval != 0 and
            if old_iteration > 10 and old_iteration < 15000 and old_iteration % opt.densification_interval != 0:
                    gaussians.load_snapshot_state(best_model)
            if old_iteration > 15000 or old_iteration ==15000:
                    gaussians.load_snapshot_state(best_model)
        else:
            print("wrong not always")

        if (old_iteration in saving_iterations):
            scene.save(old_iteration)
        
                

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
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

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set the configuration parameters on args, if they are not already set by command line arguments
        for key, value in config.items():
            setattr(args, key, value)

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
    

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seeds)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    
    
    
    training(lp.extract(args), opt, pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
