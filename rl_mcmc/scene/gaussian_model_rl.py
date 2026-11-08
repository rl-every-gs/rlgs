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
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from scene.gaussian_model import GaussianModel

from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.reloc_utils import compute_relocation_cuda
import copy
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
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
            
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
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2)*0.1)[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, reset_params=True):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if reset_params:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

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
            "shoptimizer": copy.deepcopy(self.shoptimizer.state_dict()) if self.shoptimizer else None,
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
            self.shoptimizer.load_state_dict(snapshot["shoptimizer"])

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
            self.shoptimizer.load_state_dict(snapshot["shoptimizer"])

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