"""Reinforcement-learning utilities used by the released RL-MCMC trainer.

These functions were previously imported from the legacy TMGS training script.
Keeping them here makes the public RL-MCMC path self-contained without changing
the validated policy or reward computation.
"""

import os

import torch.nn.functional as F

from gaussian_renderer import render
from utils.image_utils import psnr


lr_search_space = [
    ("xyz", 0.00016, 1e-5, 5e-4),
    ("scaling", 0.005, 1e-4, 1e-2),
    ("rotation", 0.001, 1e-4, 5e-3),
    ("opacity", 0.025, 1e-3, 5e-2),
    ("f_dc", 0.0025, 1e-4, 5e-3),
]

# Retained for checkpoint and RNG compatibility with the validated trainer.
densi_hp_space = [
    ("densify_grad_threshold", 0.0002, 0.00018, 0.00022),
    ("cameras_extent_scale", 1.0, 0.9, 1.1),
]


def psnr_reward(viewpoint_cam, gaussians, pipe, bg, dataset, gt_image):
    del dataset
    render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
    image = render_pkg["render"]
    visibility_filter = render_pkg["visibility_filter"]

    psnr_value = psnr(image, gt_image).mean()
    over_low = F.relu(-image)
    over_high = F.relu(image - 1.0)
    penalty = (over_low + over_high).mean()

    lambda_penalty = float(os.environ["lambda_penalty"])
    reward = psnr_value - lambda_penalty * penalty
    return reward, visibility_filter


def compute_policy_loss_entropy(action_dict, reward, entropy_coef=0.01):
    policy_loss = 0.0
    entropy_loss = 0.0

    for value in action_dict.values():
        policy_loss += (-value["log_prob"] * reward).mean()
        entropy_loss += value["entropy"].mean()

    return policy_loss - entropy_coef * entropy_loss
