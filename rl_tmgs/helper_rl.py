import torch
import math
import numpy as np
import random


def compute_test_psnr_reward(reward_cameras_list, num_reward_views, gaussians, lr_adjust_dict, pipe, bg, dataset, psnr_func, visibility_filter):
    gaussians.optimizer.undo() # back to last state 
    gaussians.adjustlr_by_actions_selective(lr_adjust_dict=lr_adjust_dict, mask=visibility_filter)
    gaussians.optimizer.step()
    attr_loss_rl_sum = 0
    for viewpoint_cam in reward_cameras_list:
        gt_image = viewpoint_cam.original_image.cuda()
        attr_loss_rl, visible_filter_lr = psnr_func(viewpoint_cam, gaussians, pipe, bg, dataset, gt_image)
        attr_loss_rl_sum += attr_loss_rl
    return attr_loss_rl_sum / num_reward_views # - psnr, smaller the better so reward is psoive
    

        
        
def compute_reward_mnpsnr(reward_cameras_list, num_reward_views, gaussians, lr_adjust_dict, pipe, bg, dataset, negative_psnr_loss, visibility_filter):
    
    attr_loss_sgd_sum = 0
    for viewpoint_cam in reward_cameras_list:
        gt_image = viewpoint_cam.original_image.cuda()
        attr_loss_sgd, visible_filter_sgd = negative_psnr_loss(viewpoint_cam, gaussians, pipe, bg, dataset, gt_image)
        attr_loss_sgd_sum += attr_loss_sgd
        
    gaussians.optimizer.undo() # back to last state 
    gaussians.adjustlr_by_actions_selective(lr_adjust_dict=lr_adjust_dict, mask=visibility_filter)
    gaussians.optimizer.step()
    attr_loss_rl_sum = 0
    for viewpoint_cam in reward_cameras_list:
        gt_image = viewpoint_cam.original_image.cuda()
        attr_loss_rl, visible_filter_lr = negative_psnr_loss(viewpoint_cam, gaussians, pipe, bg, dataset, gt_image)
        attr_loss_rl_sum += attr_loss_rl
    return (attr_loss_sgd_sum - attr_loss_rl_sum)/num_reward_views
        
def compute_reward_npsnr(reward_cameras_list, gaussians, lr_adjust_dict, pipe, bg, dataset, negative_psnr_loss, visibility_filter):
    
    attr_loss_sgd_sum = 0
    for viewpoint_cam in reward_cameras_list:
        gt_image = viewpoint_cam.original_image.cuda()
        attr_loss_sgd, visible_filter_sgd = negative_psnr_loss(viewpoint_cam, gaussians, pipe, bg, dataset, gt_image)
        attr_loss_sgd_sum += attr_loss_sgd
        
    gaussians.optimizer.undo() # back to last state 
    gaussians.adjustlr_by_actions_selective(lr_adjust_dict=lr_adjust_dict, mask=visibility_filter)
    gaussians.optimizer.step()
    attr_loss_rl_sum = 0
    for viewpoint_cam in reward_cameras_list:
        gt_image = viewpoint_cam.original_image.cuda()
        attr_loss_rl, visible_filter_lr = negative_psnr_loss(viewpoint_cam, gaussians, pipe, bg, dataset, gt_image)
        attr_loss_rl_sum += attr_loss_rl
    return (attr_loss_sgd_sum - attr_loss_rl_sum)
        
def compute_reward_efrac(gaussians, lr_adjust_dict, visibility_filter):
    
    attr_loss_sgd_sum = fractal_dimension_energy(gaussians._xyz)[1]
        
    gaussians.optimizer.undo() # back to last state 
    gaussians.adjustlr_by_actions_selective(lr_adjust_dict=lr_adjust_dict, mask=visibility_filter)
    gaussians.optimizer.step()
    attr_loss_rl_sum = fractal_dimension_energy(gaussians._xyz)[1]
    return 100*(attr_loss_sgd_sum - attr_loss_rl_sum)

def compute_policy_loss(lr_adjust_dict, reward):
    policy_loss = 0.0
    for key, value in lr_adjust_dict.items():
        log_prob = value["log_prob"]  #  
        policy_loss += (-log_prob * reward).mean()
    return policy_loss



def compute_policy_loss_noise(lr_adjust_dict, returns, baseline, ent_coef=1e-3):
    """
    lr_adjust_dict: 同你现在的结构, 每 key 含 {"action", "log_prob"} (shape [B,1])
    returns:        蒙特卡洛或 λ‑return, shape [B]
    baseline:       V(s) 预测值,           shape [B]  (detach 外挂 critic)
    """

    log_probs = torch.cat([v["log_prob"] for v in lr_adjust_dict.values()], dim=-1)
    log_probs = log_probs.sum(dim=-1)            # [B]


    adv = (returns - baseline).detach()

    # ③ policy gradient
    pg_loss = -(log_probs * adv).mean()


    entropy = -(log_probs.exp() * log_probs).mean()
    loss = pg_loss - ent_coef * entropy
    return loss


class ContinuousEnvLargeTorchPerPoint:
    def __init__(self, num_points=1000000, device='cpu'):
        self.num_points = num_points
        self.device = device

        self.state_dims = {
            "xyz": (self.num_points, 3),
            "opacity": (self.num_points, 1),
            "scaling": (self.num_points, 1),
            "rotations": (self.num_points, 3)
        }

        self.total_state_dim = sum(math.prod(shape) for shape in self.state_dims.values())

        self.action_dim = 8

        self.state = self.reset()

    def reset(self):
        """生成一个随机状态，返回状态字典"""
        state = {key: torch.rand(shape, device=self.device) for key, shape in self.state_dims.items()}
        self.state = state
        return state

    def step(self, actions):
        """
        根据输入动作更新状态。
        参数 actions 应为形状 (num_points, 8) 的张量，其中：
          - actions[:, :3] 用于更新 xyz
          - actions[:, 3:4] 用于更新 opacity
          - actions[:, 4:5] 用于更新 scaling
          - actions[:, 5:] 用于更新 rotations
        返回：
          - new_state：更新后的状态字典
          - reward：本次更新的奖励（这里简单地以更新幅度绝对值的和的负值作为奖励）
          - done：以5%的概率返回 True 表示回合终止
          - info：额外信息（此处为空字典）
        """

        assert actions.shape == (self.num_points, self.action_dim), "Action shape mismatch"
        lr = 0.1


        delta_xyz = actions[:, :3]          # (num_points, 3)
        delta_opacity = actions[:, 3:4]       # (num_points, 1)
        delta_scaling = actions[:, 4:5]       # (num_points, 1)
        delta_rotations = actions[:, 5:]      # (num_points, 3)

        new_state = {}
        new_state["xyz"] = self.state["xyz"] + lr * delta_xyz
        new_state["opacity"] = self.state["opacity"] + lr * delta_opacity
        new_state["scaling"] = self.state["scaling"] + lr * delta_scaling
        new_state["rotations"] = self.state["rotations"] + lr * delta_rotations


        reward = -sum(torch.abs(new_state[k] - self.state[k]).sum() for k in self.state.keys())

        done = (torch.rand(1, device=self.device).item() > 0.95)
        self.state = new_state
        return new_state, reward, done, {}

    def sample_action(self):
        """从 [-1,1] 内均匀采样一个形状为 (num_points, 8) 的动作张量"""
        actions = torch.rand(self.num_points, self.action_dim, device=self.device) * 2 - 1
        return actions

def get_grads(reward_cameras_list, gaussians, pipe, bg, dataset, render_loss):
    gradient_all = []
    for reward_viewpoint_cam in reward_cameras_list:
        # Transfer the original image to the CUDA device.
        gt_image = reward_viewpoint_cam.original_image.cuda()

        # Compute the rendering loss and any additional visibility filter.
        attr_loss_rl, visible_filter_lr = render_loss(
            reward_viewpoint_cam, gaussians, pipe, bg, dataset, gt_image
        )

        # Backpropagate the loss.
        gaussians.optimizer.zero_grad()
        attr_loss_rl.backward()

        # Update the weights using the optimizer and zero out gradients for the next iteration.
        gradient_all.append(gaussians.snapshot_gradient())
    return gradient_all
    

def try_gradient_updating(reward_cameras_list, gaussians, pipe, bg, dataset, render_loss):
    """
    Iterates over a list of reward cameras, computes the rendering loss,
    backpropagates the loss, and steps the optimizer for each camera.

    Parameters:
        reward_cameras_list (list): List of reward camera objects, each with an 
                                    `original_image` attribute that supports .cuda().
        gaussians: Object containing the optimizer (e.g., a PyTorch module with an optimizer attribute).
        pipe: A pipeline or model component used in computing the render loss.
        bg: Background parameter used within the render loss function.
        dataset: The dataset or related parameter required by the render_loss function.
    
    Returns:
        None
    """
    
    for reward_viewpoint_cam in reward_cameras_list:
        # Transfer the original image to the CUDA device.
        gt_image = reward_viewpoint_cam.original_image.cuda()

        # Compute the rendering loss and any additional visibility filter.
        
        #
        # viewpoint_cam, gaussians, pipe, bg, dataset,gt_image
        # 
        attr_loss_rl, visible_filter_lr = render_loss(
            reward_viewpoint_cam, gaussians, pipe, bg, dataset, gt_image
        )

        # Backpropagate the loss.
        gaussians.shoptimizer.zero_grad()
        gaussians.optimizer.zero_grad()
        attr_loss_rl.backward(retain_graph=True)

        # Update the weights using the optimizer and zero out gradients for the next iteration.
        gaussians.zero_momenta()
        gaussians.optimizer.step()
        gaussians.shoptimizer.step()
        
        gaussians.shoptimizer.zero_grad(set_to_none = True)
        gaussians.optimizer.zero_grad(set_to_none = True)




def try_gradient_updating_return_loss(reward_cameras_list, gaussians, pipe, bg, dataset, render_loss):
    """
    Iterates over a list of reward cameras, computes the rendering loss,
    backpropagates the loss, and steps the optimizer for each camera.

    Parameters:
        reward_cameras_list (list): List of reward camera objects, each with an 
                                    `original_image` attribute that supports .cuda().
        gaussians: Object containing the optimizer (e.g., a PyTorch module with an optimizer attribute).
        pipe: A pipeline or model component used in computing the render loss.
        bg: Background parameter used within the render loss function.
        dataset: The dataset or related parameter required by the render_loss function.
    
    Returns:
        None
    """
    
    for reward_viewpoint_cam in reward_cameras_list:
        # Transfer the original image to the CUDA device.
        gt_image = reward_viewpoint_cam.original_image.cuda()

        # Compute the rendering loss and any additional visibility filter.
        
        #
        # viewpoint_cam, gaussians, pipe, bg, dataset,gt_image
        # 
        attr_loss, visibility_filter, viewspace_point_tensor, radii = render_loss(
            reward_viewpoint_cam, gaussians, pipe, bg, dataset, gt_image
        )

        # Backpropagate the loss.
        gaussians.shoptimizer.zero_grad()
        gaussians.optimizer.zero_grad()
        attr_loss.backward(retain_graph=True)

        # Update the weights using the optimizer and zero out gradients for the next iteration.
        gaussians.zero_momenta()
        gaussians.optimizer.step()
        gaussians.shoptimizer.step()
        
        gaussians.shoptimizer.zero_grad(set_to_none = True)
        gaussians.optimizer.zero_grad(set_to_none = True)
    return attr_loss, visibility_filter, viewspace_point_tensor, radii

def try_gradient_updating_v3(reward_cameras_list, gaussians, pipe, bg, dataset, render_loss,lr_adjust_dict, lr_parameter_ori_dict, iteration):
    """
    Iterates over a list of reward cameras, computes the rendering loss,
    backpropagates the loss, and steps the optimizer for each camera.

    Parameters:
        reward_cameras_list (list): List of reward camera objects, each with an 
                                    `original_image` attribute that supports .cuda().
        gaussians: Object containing the optimizer (e.g., a PyTorch module with an optimizer attribute).
        pipe: A pipeline or model component used in computing the render loss.
        bg: Background parameter used within the render loss function.
        dataset: The dataset or related parameter required by the render_loss function.
    
    Returns:
        None
    """
    local_step = 1 
    for reward_viewpoint_cam in reward_cameras_list:
        # Transfer the original image to the CUDA device.
        gt_image = reward_viewpoint_cam.original_image.cuda()

        # Compute the rendering loss and any additional visibility filter.
        attr_loss_rl, visible_filter_lr = render_loss(
            reward_viewpoint_cam, gaussians, pipe, bg, dataset, gt_image
        )

        # Backpropagate the loss.
        
        xyz_lr = gaussians.update_learning_rate(iteration+local_step)
        lr_parameter_ori_dict['xyz'] = xyz_lr
        gaussians.adjust_lr_by_actions(lr_adjust_dict=lr_adjust_dict, parameter_name_dict=lr_parameter_ori_dict)
         
        gaussians.optimizer.zero_grad()
        attr_loss_rl.backward()

        # Update the weights using the optimizer and zero out gradients for the next iteration.
        
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad()
        local_step += 1

def gradient_compute(reward_cameras_list, gaussians, pipe, bg, dataset, render_loss):
    """
    Iterates over a list of reward cameras, computes the rendering loss,
    backpropagates the loss, and steps the optimizer for each camera.

    Parameters:
        reward_cameras_list (list): List of reward camera objects, each with an 
                                    `original_image` attribute that supports .cuda().
        gaussians: Object containing the optimizer (e.g., a PyTorch module with an optimizer attribute).
        pipe: A pipeline or model component used in computing the render loss.
        bg: Background parameter used within the render loss function.
        dataset: The dataset or related parameter required by the render_loss function.
    
    Returns:
        None
    """
    gaussians.optimizer.zero_grad()
    for reward_viewpoint_cam in reward_cameras_list:
        # Transfer the original image to the CUDA device.
        gt_image = reward_viewpoint_cam.original_image.cuda()

        # Compute the rendering loss and any additional visibility filter.
        attr_loss_rl, visible_filter_lr = render_loss(
            reward_viewpoint_cam, gaussians, pipe, bg, dataset, gt_image
        )

        # Backpropagate the loss.
        
        attr_loss_rl.backward()

        return visible_filter_lr
        
def try_gradient_updating_v2(reward_cameras_list, gaussians, pipe, bg, dataset, render_loss):
    """
    Iterates over a list of reward cameras, computes the rendering loss,
    backpropagates the loss, and steps the optimizer for each camera.

    Parameters:
        reward_cameras_list (list): List of reward camera objects, each with an 
                                    `original_image` attribute that supports .cuda().
        gaussians: Object containing the optimizer (e.g., a PyTorch module with an optimizer attribute).
        pipe: A pipeline or model component used in computing the render loss.
        bg: Background parameter used within the render loss function.
        dataset: The dataset or related parameter required by the render_loss function.
    
    Returns:
        None
    """
    loss = 0
    for reward_viewpoint_cam in reward_cameras_list:
        # Transfer the original image to the CUDA device.
        gt_image = reward_viewpoint_cam.original_image.cuda()

        # Compute the rendering loss and any additional visibility filter.
        attr_loss_rl, visible_filter_lr = render_loss(
            reward_viewpoint_cam, gaussians, pipe, bg, dataset, gt_image
        )

        # Backpropagate the loss.
        attr_loss_rl.backward()
        loss += attr_loss_rl.clone().detach()

        # Update the weights using the optimizer and zero out gradients for the next iteration.
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad()
        
    return loss 

def try_gradient_updating_densi(reward_cameras_list, gaussians, pipe, bg, dataset, render_loss):
    """
    Iterates over a list of reward cameras, computes the rendering loss,
    backpropagates the loss, and steps the optimizer for each camera.

    Parameters:
        reward_cameras_list (list): List of reward camera objects, each with an 
                                    `original_image` attribute that supports .cuda().
        gaussians: Object containing the optimizer (e.g., a PyTorch module with an optimizer attribute).
        pipe: A pipeline or model component used in computing the render loss.
        bg: Background parameter used within the render loss function.
        dataset: The dataset or related parameter required by the render_loss function.
    
    Returns:
        None
    """
    for reward_viewpoint_cam in reward_cameras_list:
        # Transfer the original image to the CUDA device.
        gt_image = reward_viewpoint_cam.original_image.cuda()

        # Compute the rendering loss and any additional visibility filter.
        attr_loss, visibility_filter, viewspace_point_tensor, radii = render_loss(
            reward_viewpoint_cam, gaussians, pipe, bg, dataset, gt_image
        )

        # Backpropagate the loss.
        gaussians.optimizer.zero_grad()
        gaussians.shoptimizer.zero_grad()
        attr_loss.backward()

        # Update the weights using the optimizer and zero out gradients for the next iteration.
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad()
        
        gaussians.shoptimizer.step()
        gaussians.shoptimizer.zero_grad(set_to_none = True)
        # Keep track of max radii in image-space for pruning
        #gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        #gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
    return viewspace_point_tensor, visibility_filter, radii # radii for densifciation
        
        



def try_gradient_updating_rl(reward_cameras_list, gaussians, pipe, bg, dataset, render_loss, lr_dict):
    """
    Iterates over a list of reward cameras, computes the rendering loss,
    backpropagates the loss, and steps the optimizer for each camera.

    Parameters:
        reward_cameras_list (list): List of reward camera objects, each with an 
                                    `original_image` attribute that supports .cuda().
        gaussians: Object containing the optimizer (e.g., a PyTorch module with an optimizer attribute).
        pipe: A pipeline or model component used in computing the render loss.
        bg: Background parameter used within the render loss function.
        dataset: The dataset or related parameter required by the render_loss function.
    
    Returns:
        None
    """
    for reward_viewpoint_cam in reward_cameras_list:
        # Transfer the original image to the CUDA device.
        gt_image = reward_viewpoint_cam.original_image.cuda()

        # Compute the rendering loss and any additional visibility filter.
        attr_loss_rl, visible_filter_lr = render_loss(
            reward_viewpoint_cam, gaussians, pipe, bg, dataset, gt_image
        )

        # Backpropagate the loss.
        attr_loss_rl.backward()
        with torch.no_grad():
            gaussians.adjustlr_by_actions_selective(lr_dict, visible_filter_lr)

        # Update the weights using the optimizer and zero out gradients for the next iteration.
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad()


def evaluate_total_gain_psnr(reward_cameras_list, gaussians, pipe, bg, dataset, psnr_reward):
    """
    Computes the total PSNR by summing the PSNR values for each reward camera.

    Parameters:
        reward_cameras_list (list): List of reward camera objects, each having an 
                                    `original_image` attribute that can be moved to CUDA.
        gaussians: Object used in the computation of PSNR (e.g., a model or parameter holder).
        pipe: A model or pipeline component used for PSNR computation.
        bg: Background parameter for the PSNR computation.
        dataset: The dataset or related parameter required by the psnr_reward function.
    
    Returns:
        float: The accumulated PSNR from all reward camera images.
    """
    with torch.no_grad():
        sum_psnr = 0 
        for reward_viewpoint_cam in reward_cameras_list:
            # Move the original image to CUDA.
            gt_image = reward_viewpoint_cam.original_image.cuda()
            # Compute the PSNR for the camera using the provided psnr_reward function.
            tmp_psnr = psnr_reward(reward_viewpoint_cam, gaussians, pipe, bg, dataset, gt_image)[0]
            sum_psnr += tmp_psnr
        return sum_psnr

def evaluate_total_gain_psnr_vis(reward_viewpoint_cam, gaussians, pipe, bg, dataset, psnr_reward):
    """
    Computes the total PSNR by summing the PSNR values for each reward camera.

    Parameters:
        reward_cameras_list (list): List of reward camera objects, each having an 
                                    `original_image` attribute that can be moved to CUDA.
        gaussians: Object used in the computation of PSNR (e.g., a model or parameter holder).
        pipe: A model or pipeline component used for PSNR computation.
        bg: Background parameter for the PSNR computation.
        dataset: The dataset or related parameter required by the psnr_reward function.
    
    Returns:
        float: The accumulated PSNR from all reward camera images.
    """
    with torch.no_grad():
        sum_psnr = 0 
        
            # Move the original image to CUDA.
        gt_image = reward_viewpoint_cam.original_image.cuda()
        # Compute the PSNR for the camera using the provided psnr_reward function.
        rendered_image = psnr_reward(reward_viewpoint_cam, gaussians, pipe, bg, dataset, gt_image)[0]
        return rendered_image  
      



def evaluate_total_gain_psnr_minmax(reward_cameras_list, gaussians, pipe, bg, dataset, psnr_reward):
    """
    Computes the total PSNR by summing the PSNR values for each reward camera.

    Parameters:
        reward_cameras_list (list): List of reward camera objects, each having an 
                                    `original_image` attribute that can be moved to CUDA.
        gaussians: Object used in the computation of PSNR (e.g., a model or parameter holder).
        pipe: A model or pipeline component used for PSNR computation.
        bg: Background parameter for the PSNR computation.
        dataset: The dataset or related parameter required by the psnr_reward function.
    
    Returns:
        float: The accumulated PSNR from all reward camera images.
    """
    with torch.no_grad():
        sum_psnr = 0 
        total = 0.0
        count = 0
        max_val = float('-inf')
        min_val = float('inf')


        for reward_viewpoint_cam in reward_cameras_list:
            # Move the original image to CUDA.
            gt_image = reward_viewpoint_cam.original_image.cuda()
            # Compute the PSNR for the camera using the provided psnr_reward function.
            tmp_psnr = psnr_reward(reward_viewpoint_cam, gaussians, pipe, bg, dataset, gt_image)[0]
            sum_psnr += tmp_psnr
            
            if tmp_psnr > max_val:
                max_val = tmp_psnr
            if tmp_psnr < min_val:
                min_val = tmp_psnr
            
        return sum_psnr - max_val - min_val
def evaluate_total_gain_psnr_rt_list(reward_cameras_list, gaussians, pipe, bg, dataset, psnr_reward):
    """
    Computes the total PSNR by summing the PSNR values for each reward camera.

    Parameters:
        reward_cameras_list (list): List of reward camera objects, each having an 
                                    `original_image` attribute that can be moved to CUDA.
        gaussians: Object used in the computation of PSNR (e.g., a model or parameter holder).
        pipe: A model or pipeline component used for PSNR computation.
        bg: Background parameter for the PSNR computation.
        dataset: The dataset or related parameter required by the psnr_reward function.
    
    Returns:
        float: The accumulated PSNR from all reward camera images.
    """
    with torch.no_grad():
        sum_psnr = []
        for reward_viewpoint_cam in reward_cameras_list:
            # Move the original image to CUDA.
            gt_image = reward_viewpoint_cam.original_image.cuda()
            # Compute the PSNR for the camera using the provided psnr_reward function.
            tmp_psnr = psnr_reward(reward_viewpoint_cam, gaussians, pipe, bg, dataset, gt_image)[0]
            sum_psnr.append(tmp_psnr.detach())
        return sum_psnr



def get_psnr_dict(reward_cameras_list, gaussians, pipe, bg, dataset, psnr_reward):
    """
    Computes the total PSNR by summing the PSNR values for each reward camera.

    Parameters:
        reward_cameras_list (list): List of reward camera objects, each having an 
                                    `original_image` attribute that can be moved to CUDA.
        gaussians: Object used in the computation of PSNR (e.g., a model or parameter holder).
        pipe: A model or pipeline component used for PSNR computation.
        bg: Background parameter for the PSNR computation.
        dataset: The dataset or related parameter required by the psnr_reward function.
    
    Returns:
        float: The accumulated PSNR from all reward camera images.
    """
    tmp = {}
    with torch.no_grad():
        for reward_viewpoint_cam in reward_cameras_list:
            # Move the original image to CUDA.
            gt_image = reward_viewpoint_cam.original_image.cuda()
            # Compute the PSNR for the camera using the provided psnr_reward function.
            tmp_psnr = psnr_reward(reward_viewpoint_cam, gaussians, pipe, bg, dataset, gt_image)[0]
            tmp[reward_viewpoint_cam.image_name] = tmp_psnr
        return tmp


def evaluate_train_psnr(reward_cameras_list, gaussians, pipe, bg, dataset, psnr_reward):
    """
    Computes the total PSNR by summing the PSNR values for each reward camera.

    Parameters:
        reward_cameras_list (list): List of reward camera objects, each having an 
                                    `original_image` attribute that can be moved to CUDA.
        gaussians: Object used in the computation of PSNR (e.g., a model or parameter holder).
        pipe: A model or pipeline component used for PSNR computation.
        bg: Background parameter for the PSNR computation.
        dataset: The dataset or related parameter required by the psnr_reward function.
    
    Returns:
        float: The accumulated PSNR from all reward camera images.
    """
    sum_psnr = 0 
    for reward_viewpoint_cam in reward_cameras_list:
        # Move the original image to CUDA.
        gt_image = reward_viewpoint_cam.original_image.cuda()
        # Compute the PSNR for the camera using the provided psnr_reward function.
        tmp_psnr = psnr_reward(reward_viewpoint_cam, gaussians, pipe, bg, dataset, gt_image)[0]
        sum_psnr += tmp_psnr
    return sum_psnr


def evaluate_total_gain_psnr_list(reward_cameras_list, gaussians, pipe, bg, dataset, psnr_reward):
    """
    Computes the total PSNR by summing the PSNR values for each reward camera.

    Parameters:
        reward_cameras_list (list): List of reward camera objects, each having an 
                                    `original_image` attribute that can be moved to CUDA.
        gaussians: Object used in the computation of PSNR (e.g., a model or parameter holder).
        pipe: A model or pipeline component used for PSNR computation.
        bg: Background parameter for the PSNR computation.
        dataset: The dataset or related parameter required by the psnr_reward function.
    
    Returns:
        float: The accumulated PSNR from all reward camera images.
    """
    sum_psnr = [] 
    for reward_viewpoint_cam in reward_cameras_list:
        # Move the original image to CUDA.
        gt_image = reward_viewpoint_cam.original_image.cuda()
        # Compute the PSNR for the camera using the provided psnr_reward function.
        tmp_psnr = psnr_reward(reward_viewpoint_cam, gaussians, pipe, bg, dataset, gt_image)[0]
        sum_psnr.append(tmp_psnr)
    return sum_psnr

if __name__ == '__main__':
    env = ContinuousEnvLargeTorchPerPoint(device="cuda")
    state = env.reset()
    print("状态键:", list(state.keys()))
    print("xyz的形状:", state["xyz"].shape)
    print("整体状态扁平化后的总维度:", env.total_state_dim)
    
    actions = env.sample_action()
    print("采样动作的形状:", actions.shape)
    
    next_state, reward, done, _ = env.step(actions)
    print("奖励:", reward.item(), "终止:", done)



def change_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(torch.device("cuda:0"))
