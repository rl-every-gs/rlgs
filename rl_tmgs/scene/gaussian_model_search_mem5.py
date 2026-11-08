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
from helper_optimzer import AdamUndo
import copy


def _nested_tensors_to_device(obj, device):
    """Recursively move Torch tensors in nested dict/list to *device* without mutating CPU copies."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _nested_tensors_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nested_tensors_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_nested_tensors_to_device(v, device) for v in obj)
    return obj


def _nested_tensors_to_cpu(obj):
    """Recursively copy tensors to CPU without keeping an intermediate full GPU clone tree."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _nested_tensors_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nested_tensors_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_nested_tensors_to_cpu(v) for v in obj)
    return obj


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


    def __init__(self, sh_degree, opt, lr_policy_network=None, densi_policy_network=None):
        self.active_sh_degree = 0
        self.optimizer_type = opt.optimizer_type
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
                        #param_group['lr'] =  float(lr_adjust_dict["f_dc"]['action']) * parameter_name_dict["f_dc"] / 20.0
    
    
    
    def adjust_grad_by_spatial_actions(
        self,
        spatial_action_dict,
        gaussian_cell_idx,     # dict: {"ix":, "iy":, "iz":} or tuple
    ):
        """
        spatial_action_dict:
            out = policy(...)
            {
                "xyz": {"ax","ay","az"},
                "scaling": {...},
                ...
            }

        gaussian_cell_idx:
            ix, iy, iz: [P]
        """
        ix, iy, iz = gaussian_cell_idx

        with torch.no_grad():
            for key, params in self.named_parameters_dict.items():
                # params: Tensor grad holder, e.g. means, scales, rotations...
                if params.grad is None:
                    continue

                if key not in spatial_action_dict:
                    continue

                # build spatial weight
                w_grid = self.spatial_policy_network.build_weight_grid(
                    spatial_action_dict, key
                )  # [Nx,Ny,Nz]

                w = w_grid[ix, iy, iz]  # [P]


                if params.grad.dim() == 2:        # [P, C]
                    params.grad.mul_(w.unsqueeze(-1))
                elif params.grad.dim() == 1:      # [P]
                    params.grad.mul_(w)
                else:
                    raise ValueError(f"Unsupported grad shape for {key}")

    
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
    def save_policy(self,path):
        torch.save(self.lr_policy_network, path.replace("point_cloud.ply", "grd_policy_network.torch"))
        torch.save(self.densi_policy_network, path.replace("point_cloud.ply", "densi_policy_network.torch"))

    def _state_device(self) -> torch.device:
        """Gaussian tensors are plain Tensors on this class (not nn.Module parameters)."""
        xyz = getattr(self, "_xyz", None)
        if isinstance(xyz, torch.Tensor) and xyz.numel() > 0:
            return xyz.device
        if self.optimizer is not None:
            for g in self.optimizer.param_groups:
                params = g.get("params")
                if params:
                    return params[0].device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def snapshot_state(
        self,
        include_grads: bool = True,
        include_policy: bool = False,
        store_on_cpu: bool = False,
    ) -> dict:
        """
        Return a *deep* dictionary containing everything required to bring the
        model back to this exact step.

        Args
        ----
        include_grads : if True, store .grad tensors as well.
        include_policy: if True and a policy network exists, also snapshot
                        its weights + optimiser.
        store_on_cpu : if True, copy tensors straight to CPU (avoids a full GPU
                       clone peak); snapshots are sized for `_nested_tensors_to_device` on load.
        """
        if store_on_cpu:
            def _tensor_snap(t):
                return t.detach().cpu().clone()
        else:
            def _tensor_snap(t):
                return t.detach().clone()

        snap = {
            "densi": {"xyz_gradient_accum": _tensor_snap(self.xyz_gradient_accum),
                      "denom": _tensor_snap(self.denom),
                      "max_radii2D": _tensor_snap(self.max_radii2D)},
            "meta": {
                "active_sh_degree": self.active_sh_degree,
                "spatial_lr_scale": self.spatial_lr_scale,
            },
            "tensors": {
                "_xyz":           _tensor_snap(self._xyz),
                "_features_dc":   _tensor_snap(self._features_dc),
                "_features_rest": _tensor_snap(self._features_rest),
                "_opacity":       _tensor_snap(self._opacity),
                "_scaling":       _tensor_snap(self._scaling),
                "_rotation":      _tensor_snap(self._rotation),
            },
            "optimizer": (_nested_tensors_to_cpu(self.optimizer.state_dict())
                         if store_on_cpu and self.optimizer else None),
            "shoptimizer": (_nested_tensors_to_cpu(self.shoptimizer.state_dict())
                           if store_on_cpu and self.shoptimizer else None),
            "exposure_optimizer": (_nested_tensors_to_cpu(self.exposure_optimizer.state_dict())
                                 if store_on_cpu and hasattr(self, "exposure_optimizer")
                                 and self.exposure_optimizer else None),
        }
        if not store_on_cpu:
            snap["optimizer"] = (
                copy.deepcopy(self.optimizer.state_dict()) if self.optimizer else None)
            snap["shoptimizer"] = (
                copy.deepcopy(self.shoptimizer.state_dict()) if self.shoptimizer else None)
            snap["exposure_optimizer"] = (
                copy.deepcopy(self.exposure_optimizer.state_dict())
                if hasattr(self, "exposure_optimizer") and self.exposure_optimizer else None)

        # Optional exposure tensor
        if hasattr(self, "_exposure"):
            snap["tensors"]["_exposure"] = _tensor_snap(self._exposure)

        # Optional gradients
        if include_grads:
            if store_on_cpu:
                snap["grad"] = {
                    name: (_nested_tensors_to_cpu(getattr(self, name).grad)
                           if getattr(self, name).grad is not None else None)
                    for name in snap["tensors"]
                }
            else:
                snap["grad"] = {
                    name: (getattr(self, name).grad.clone()
                           if getattr(self, name).grad is not None else None)
                    for name in snap["tensors"]
                }

        # Optional policy network
        if include_policy and self.grd_policy_network is not None:
            if store_on_cpu:
                snap["lr_policy_network"] = _nested_tensors_to_cpu(
                    self.grd_policy_network.state_dict())
                snap["lr_policy_optimizer"] = (
                    _nested_tensors_to_cpu(self.grd_policy_optimizer.state_dict())
                    if self.grd_policy_optimizer else None)
            else:
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
        _dev = self._state_device()
        # ----- meta -----
        self.active_sh_degree = snapshot["meta"]["active_sh_degree"]
        self.spatial_lr_scale = snapshot["meta"]["spatial_lr_scale"]
        self.denom = snapshot["densi"]["denom"].to(self.denom.device)
        self.xyz_gradient_accum = snapshot["densi"]["xyz_gradient_accum"].to(self.xyz_gradient_accum.device)
        self.max_radii2D = snapshot["densi"]["max_radii2D"].to(self.max_radii2D.device)
        # ----- tensors -----
        for name, tensor in snapshot["tensors"].items():
            if not hasattr(self, name):
                continue  # e.g. _exposure may not exist in some configs
            #getattr(self, name).data.copy_(tensor)
            target = getattr(self, name)
            snap_t = tensor.to(target.device)
            with torch.no_grad():
                target.copy_(snap_t)
            

        # ----- optimiser states (CPU snapshots -> GPU) -----
        if self.optimizer is not None and snapshot["optimizer"] is not None:
            self.optimizer.load_state_dict(_nested_tensors_to_device(snapshot["optimizer"], _dev))
        if self.shoptimizer is not None and snapshot.get("shoptimizer") is not None:
            self.shoptimizer.load_state_dict(_nested_tensors_to_device(snapshot["shoptimizer"], _dev))

        if (hasattr(self, "exposure_optimizer") and self.exposure_optimizer is not None
                and snapshot.get("exposure_optimizer") is not None):
            _e_dev = _dev
            for g in self.exposure_optimizer.param_groups:
                if g.get("params"):
                    _e_dev = g["params"][0].device
                    break
            self.exposure_optimizer.load_state_dict(
                _nested_tensors_to_device(snapshot["exposure_optimizer"], _e_dev))

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
                    p = getattr(self, name)
                    if g is not None:
                        p.grad = g.to(p.device, non_blocking=True)
                    else:
                        p.grad = None
                    
    
    
    
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
        _dev = self._state_device()
        # ----- meta -----
        self.active_sh_degree = snapshot["meta"]["active_sh_degree"]
        self.spatial_lr_scale = snapshot["meta"]["spatial_lr_scale"]
        self.denom = snapshot["densi"]["denom"].to(self.denom.device)
        self.xyz_gradient_accum = snapshot["densi"]["xyz_gradient_accum"].to(self.xyz_gradient_accum.device)
        self.max_radii2D = snapshot["densi"]["max_radii2D"].to(self.max_radii2D.device)
        # ----- tensors -----
        for name, tensor in snapshot["tensors"].items():
            if not hasattr(self, name):
                continue  # e.g. _exposure may not exist in some configs
            #getattr(self, name).data.copy_(tensor)
            target = getattr(self, name)
            new_tensor = tensor.detach().to(target.device, non_blocking=True).clone()
            setattr(target, "data", new_tensor)

        # # ----- rebuild opt if requested -----
        # if training_args is not None:
        #     self.training_setup(training_args)

        # ----- optimiser states (CPU snapshots -> GPU) -----
        if self.optimizer is not None and snapshot["optimizer"] is not None:
            self.optimizer.load_state_dict(_nested_tensors_to_device(snapshot["optimizer"], _dev))
        if self.shoptimizer is not None and snapshot.get("shoptimizer") is not None:
            self.shoptimizer.load_state_dict(_nested_tensors_to_device(snapshot["shoptimizer"], _dev))

        if (hasattr(self, "exposure_optimizer") and self.exposure_optimizer is not None
                and snapshot.get("exposure_optimizer") is not None):
            _e_dev = _dev
            for g in self.exposure_optimizer.param_groups:
                if g.get("params"):
                    _e_dev = g["params"][0].device
                    break
            self.exposure_optimizer.load_state_dict(
                _nested_tensors_to_device(snapshot["exposure_optimizer"], _e_dev))

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
                    p = getattr(self, name)
                    if g is not None:
                        p.grad = g.to(p.device, non_blocking=True)
                    else:
                        p.grad = None
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

            return torch.zeros(1, device=self._state_device())
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