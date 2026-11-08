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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.gaussian_model import GaussianModel
import copy

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class SGaussianModelMem2(GaussianModel):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        self.os = 1.0
        self.sr = 1.0


    def __init__(self, sh_degree, opt, lr_policy_network=None, densi_policy_network=None):
        self.active_sh_degree = 0
        
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        
        # function for the policy net
        self.lr_policy_network = lr_policy_network
        self.lr_policy_optimizer = None 
        self.densi_policy_network = densi_policy_network 
        self.densi_policy_optimizer = None 
        self.policy_lr_scheduler = None 
        self.policy_densi_scheduler = None
        
        self.attributes_list = None
        self.attributes_list_rest = None 
        self.rl_xyz = opt.rl_xyz 
        self.cache = {}  # cached state ofor other
        self._cached_state = {} # cached state for tensor 
        
        keys = ["_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest"]
        for k in keys: 
            self.cache[k] = torch.empty(0) 
 
        

        
    def training_rl_setup(self, opt, attribute_list, attribute_list_rest):
        
        grd_l = [{'params': list(self.lr_policy_network.parameters()), 'lr': opt.pl_lr, "name": "pl"}]

        self.lr_policy_optimizer = torch.optim.Adam(grd_l, lr=0.0, eps=1e-15)
        self.lr_policy_network.cuda()
        self.attributes_list = attribute_list
        self.attributes_list_rest = attribute_list_rest 
        
        densi_l = [{'params': list(self.densi_policy_network.parameters()), 'lr':opt.pl_lr, "name": "densify lr"}]
        self.densi_policy_optimizer = torch.optim.Adam(densi_l, lr=0.0, eps=1e-15)
        self.densi_policy_network.cuda()
    
    def update_learning_rate_policy(self, iteration):
        ''' Learning rate scheduling per step '''
        # if self.pretrained_exposures is None:
        #     for param_group in self.exposure_optimizer.param_groups:
        #         param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.lr_policy_optimizer.param_groups:
            lr = self.policy_lr_scheduler(iteration)
            param_group['lr'] = lr
            #break 
            #return lr
        
        
        for param_group in self.densi_policy_optimizer.param_groups:
            lr = self.policy_densi_scheduler(iteration)
            param_group['lr'] = lr
            #break
            #return lr
        return lr 
    def adjust_lr_by_actions(self, lr_adjust_dict, parameter_name_dict, step=None):
        # for k in parameter_name_dict:
        #     parameter_name_dict[k] = parameter_name_dict[k] * lr_adjust_dict[k]['action']
        with torch.no_grad():
            for param_group in self.optimizer.param_groups:
                if param_group["name"] in lr_adjust_dict.keys():
                    if param_group["name"] != "f_rest":
                        if self.rl_xyz or param_group["name"] != "xyz":
                            param_group['lr'] = float(lr_adjust_dict[param_group["name"]]['action']) * parameter_name_dict[param_group["name"]]
                    else:  
                        pass 
                        
    def adjust_lr_by_actions_v2(self, lr_adjust_dict, parameter_name_dict, step=None):
        # for k in parameter_name_dict:
        #     parameter_name_dict[k] = parameter_name_dict[k] * lr_adjust_dict[k]['action']
        with torch.no_grad():
            for param_group in self.optimizer.param_groups:
                if param_group["name"] in lr_adjust_dict.keys():
                    if param_group["name"] != "f_rest":
                        if self.rl_xyz or param_group["name"] != "xyz":
                            param_group['lr'] = float(lr_adjust_dict[param_group["name"]]['action']) * parameter_name_dict[param_group["name"]]
                    else:  
                        pass 
        self.os = lr_adjust_dict["os"]['action']
        self.sr = lr_adjust_dict["sr"]['action']
    def restore_default_lr(self, parameter_name_dict):
        with torch.no_grad():
            for param_group in self.optimizer.param_groups:
                if param_group["name"] in parameter_name_dict:
                    if param_group["name"] != "f_rest":
                        param_group['lr'] =  parameter_name_dict[param_group["name"]]
                    else:
                        pass 
                        #param_group['lr'] =  parameter_name_dict["f_dc"] / 20
            return 
        self.os = 1.0
        self.sr = 1.0
        
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        torch.save(self.lr_policy_network, path.replace("point_cloud.ply", "grd_policy_network.torch"))
        torch.save(self.densi_policy_network, path.replace("point_cloud.ply", "densi_policy_network.torch"))

        
    def snapshot_state(
        self,
        include_grads: bool = True,
        include_policy: bool = False,
    ) -> dict:
        """
        Return a *deep* dictionary containing everything required to bring the
        model back to this exact step.  Nothing is moved off‑GPU.

        Args
        ----
        include_grads : if True, store .grad tensors as well.
        include_policy: if True and a policy network exists, also snapshot
                        its weights + optimiser.
        """
        # Helper to clone a tensor that may be an nn.Parameter
        def _clone(t):
            return t.detach().clone()

        snap = {
            "densi": {"xyz_gradient_accum": _clone(self.xyz_gradient_accum),
                      "denom":_clone(self.denom),
                      "max_radii2D":_clone(self.max_radii2D)}
            ,
            "meta": {
                "active_sh_degree": self.active_sh_degree,
                "spatial_lr_scale": self.spatial_lr_scale,
            },
            "tensors": {
                "_xyz":           _clone(self._xyz),
                "_features_dc":   _clone(self._features_dc),
                "_features_rest": _clone(self._features_rest),
                "_opacity":       _clone(self._opacity),
                "_scaling":       _clone(self._scaling),
                "_rotation":      _clone(self._rotation),
            },
            "optimizer": copy.deepcopy(self.optimizer.state_dict()) if self.optimizer else None,
            "exposure_optimizer": copy.deepcopy(self.exposure_optimizer.state_dict()) if hasattr(self, "exposure_optimizer") and self.exposure_optimizer else None,
        }

        # Optional exposure tensor
        if hasattr(self, "_exposure"):
            snap["tensors"]["_exposure"] = _clone(self._exposure)

        # Optional gradients
        if include_grads:
            snap["grad"] = {
                name: (getattr(self, name).grad.clone()
                       if getattr(self, name).grad is not None else None)
                for name in snap["tensors"]
            }

        # Optional policy network
        if include_policy and self.grd_policy_network is not None:
            snap["lr_policy_network"] = copy.deepcopy(self.grd_policy_network.state_dict())
            snap["lr_policy_optimizer"] = (
                copy.deepcopy(self.grd_policy_optimizer.state_dict())
                if self.grd_policy_optimizer else None
            )
            
        

        return snap

        
    def load_snapshot_state(
        self,
        snapshot: dict,
        restore_grads: bool = True,
        restore_policy: bool = False,
    ):
        """
        Bring the model (and optimisers) back to a snapshot produced by
        `snapshot_state`.

        If `training_args` is supplied we first rebuild the optimisers /
        schedulers with `training_setup` so the parameter lists match, then
        inject the saved state‑dicts.
        """
        # ----- meta -----
        self.active_sh_degree = snapshot["meta"]["active_sh_degree"]
        self.spatial_lr_scale = snapshot["meta"]["spatial_lr_scale"]
        self.denom = snapshot["densi"]["denom"]
        self.xyz_gradient_accum = snapshot["densi"]["xyz_gradient_accum"]
        self.max_radii2D  = snapshot["densi"]["max_radii2D"] # ":_clone(self.max_radii2D)
        # ----- tensors -----
        for name, tensor in snapshot["tensors"].items():
            if not hasattr(self, name):
                continue  # e.g. _exposure may not exist in some configs
            #getattr(self, name).data.copy_(tensor)
            target = getattr(self, name)
            snap_t = tensor.to(target.device)
            with torch.no_grad():
                target.copy_(snap_t)
            
            #new_tensor = tensor.clone()
            #setattr(getattr(self, name), "data", new_tensor)

        # # ----- rebuild opt if requested -----
        # if training_args is not None:
        #     self.training_setup(training_args)

        # ----- optimiser states -----
        if self.optimizer is not None and snapshot["optimizer"] is not None:
            self.optimizer.load_state_dict(snapshot["optimizer"])

        if (hasattr(self, "exposure_optimizer") and self.exposure_optimizer is not None
                and snapshot["exposure_optimizer"] is not None):
            self.exposure_optimizer.load_state_dict(snapshot["exposure_optimizer"])

        # ----- policy network -----
        if restore_policy and self.lr_policy_network is not None:
            if "grd_policy_network" in snapshot:
                self.lr_policy_network.load_state_dict(snapshot["grd_policy_network"])
            if (self.lr_policy_optimizer is not None
                    and "grd_policy_optimizer" in snapshot
                    and snapshot["grd_policy_optimizer"] is not None):
                self.policy_optimizer.load_state_dict(snapshot["grd_policy_optimizer"])

        # ----- gradients (rarely needed, but handy for RL roll‑back) -----
        if restore_grads and "grad" in snapshot:
            for name, g in snapshot["grad"].items():
                if hasattr(self, name):
                    setattr(getattr(self, name), "grad", g)
                    
    
    
    
    def load_snapshot_state_densi(
        self,
        snapshot: dict,
        restore_grads: bool = True,
        restore_policy: bool = False,
    ):
        """
        Bring the model (and optimisers) back to a snapshot produced by
        `snapshot_state`.

        If `training_args` is supplied we first rebuild the optimisers /
        schedulers with `training_setup` so the parameter lists match, then
        inject the saved state‑dicts.
        """
        # ----- meta -----
        self.active_sh_degree = snapshot["meta"]["active_sh_degree"]
        self.spatial_lr_scale = snapshot["meta"]["spatial_lr_scale"]
        self.denom = snapshot["densi"]["denom"]
        self.xyz_gradient_accum = snapshot["densi"]["xyz_gradient_accum"]
        self.max_radii2D  = snapshot["densi"]["max_radii2D"] # ":_clone(self.max_radii2D)
        # ----- tensors -----
        for name, tensor in snapshot["tensors"].items():
            if not hasattr(self, name):
                continue  # e.g. _exposure may not exist in some configs
            #getattr(self, name).data.copy_(tensor)
      
            new_tensor = tensor.detach().clone()
            setattr(getattr(self, name), "data", new_tensor)

        # # ----- rebuild opt if requested -----
        # if training_args is not None:
        #     self.training_setup(training_args)

        # ----- optimiser states -----
        if self.optimizer is not None and snapshot["optimizer"] is not None:
            self.optimizer.load_state_dict(snapshot["optimizer"])

        if (hasattr(self, "exposure_optimizer") and self.exposure_optimizer is not None
                and snapshot["exposure_optimizer"] is not None):
            self.exposure_optimizer.load_state_dict(snapshot["exposure_optimizer"])

        # ----- policy network -----
        if restore_policy and self.lr_policy_network is not None:
            if "grd_policy_network" in snapshot:
                self.lr_policy_network.load_state_dict(snapshot["grd_policy_network"])
            if (self.lr_policy_optimizer is not None
                    and "grd_policy_optimizer" in snapshot
                    and snapshot["grd_policy_optimizer"] is not None):
                self.policy_optimizer.load_state_dict(snapshot["grd_policy_optimizer"])

        # ----- gradients (rarely needed, but handy for RL roll‑back) -----
        if restore_grads and "grad" in snapshot:
            for name, g in snapshot["grad"].items():
                if hasattr(self, name):
                    setattr(getattr(self, name), "grad", g)
    def snapshot_gradient(self) -> dict:
        """
        Return a dict of all model tensors' .grad clones.
        If a tensor has no grad, its entry will be None.
        """
        def _clone_grad(tensor):
            return tensor.grad.clone() if tensor.grad is not None else None

        #  
        grad_fields = [
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_opacity",
            "_scaling",
            "_rotation",
        ]
        #  
        if hasattr(self, "_exposure"):
            grad_fields.append("_exposure")

        #  
        grads_snap = {
            name: _clone_grad(getattr(self, name))
            for name in grad_fields
        }
        return grads_snap
    
    def get_grad_state_vector(self) -> torch.Tensor:
        """
        直接将上面 snapshot_state() 的所有非 None grad
        flatten 后 cat，得到一个超长的一维向量。
        """
        grads = self.snapshot_state()
        vecs = []
        for name, g in grads.items():
            if g is None:

                # zero = torch.zeros_like(getattr(self, name)).flatten()
                # vecs.append(zero)
                continue
            vecs.append(g.flatten())
        if not vecs:

            return torch.zeros(1, device=next(self.parameters()).device)
        return torch.cat(vecs, dim=0)
    
    
    def check_num_g_consistent(self, iteration=0):
        num_g = self._xyz.shape[0] 
        for k in ["max_radii2D","denom", "xyz_gradient_accum"]:
            tensor = getattr(self, k)
            num_this = tensor.shape[0]
            if num_g != num_this:
                print(f"wrong densify {k}")
            if not tensor.is_cuda:
                print(f"wrong cuda {k}")

        for k in ["_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest"]:
            tensor = getattr(self, k)
            num_this = tensor.shape[0]
            
                
            if num_g != num_this:
                print(f"wrong attribute {k}")
            if not tensor.is_cuda:
                print(f"wrong cuda {k}")
            tensor = getattr(getattr(self, k), "grad")
            if tensor is not None :
                num_this = tensor.shape[0]
                if num_g != num_this:
                    print(f"wrong grad {k}")
                if not tensor.is_cuda:
                    print(f"wrong cuda {k}")
        #print(f"passed {num_g} at {iteration}")