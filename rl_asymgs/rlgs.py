"""RLGS hold-out port with complete rollout snapshots and post-densify trials.

RL-AsymGS (EMA-GS + RLGS) was developed and evaluated with:
  Python  3.11
  PyTorch 2.13.0+cu126
  torchvision 0.28.0+cu126
  CUDA    12.6
"""

import copy
import math
import random
import warnings

import torch
import torch.nn as nn

from gaussian_renderer import render
from utils.general_utils import get_expon_lr_func

# Pinned stack for the RobustNeRF Android RL-AsymGS experiments.
RLGS_TORCH_VERSION = "2.13.0"
RLGS_TORCHVISION_VERSION = "0.28.0"
RLGS_CUDA_VERSION = "12.6"
RLGS_PYTHON_VERSION = "3.11"


def _check_rlgs_torch_version():
    ver = torch.__version__.split("+")[0]
    if not ver.startswith(RLGS_TORCH_VERSION):
        warnings.warn(
            f"RL-AsymGS expected PyTorch {RLGS_TORCH_VERSION}+cu{RLGS_CUDA_VERSION.replace('.', '')}, "
            f"got {torch.__version__}. Results may differ.",
            UserWarning,
            stacklevel=2,
        )


LR_KEYS = ("xyz", "scaling", "rotation", "opacity", "f_dc")
LR_LO = torch.tensor([1e-5 / 0.00016, 1e-4 / 0.005, 1e-4 / 0.001, 1e-3 / 0.025, 1e-4 / 0.0025])
LR_HI = torch.tensor([5e-4 / 0.00016, 1e-2 / 0.005, 5e-3 / 0.001, 5e-2 / 0.025, 5e-3 / 0.0025])
DS_LO = torch.tensor([0.00018 / 0.0002, 0.9])
DS_HI = torch.tensor([0.00022 / 0.0002, 1.1])
GAUSSIAN_PARAMS = ("_xyz", "_features_dc", "_features_rest", "_scaling",
                   "_rotation", "_opacity", "_local_encoding")


class KeywiseGRUPolicy(nn.Module):
    def __init__(self, action_dim, hidden=16, init_log_std=0.0, ema_scale=5.0, max_steps=30000):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_size = hidden
        self.ema_scale = float(ema_scale)
        self.max_steps = float(max_steps)
        self.mu_base = nn.Parameter(torch.ones(action_dim))
        self.log_sigma_base = nn.Parameter(torch.ones(action_dim) * float(init_log_std))
        self.state_nets = nn.ModuleList([nn.GRUCell(2, hidden) for _ in range(action_dim)])
        self.out_nets = nn.ModuleList([nn.Linear(hidden, 2) for _ in range(action_dim)])
        for cell in self.state_nets:
            for name, param in cell.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
        for head in self.out_nets:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.hidden = [None] * action_dim

    def reset(self):
        device = self.mu_base.device
        self.hidden = [torch.zeros(1, self.hidden_size, device=device) for _ in range(self.action_dim)]

    def _state_in(self, loss_val, step):
        device = self.mu_base.device
        step_norm = torch.tensor([step / self.max_steps], dtype=torch.float32, device=device).clamp(0.0, 1.0)
        loss_t = torch.tensor([loss_val], dtype=torch.float32, device=device)
        loss_norm = torch.tanh(torch.log1p(loss_t) * self.ema_scale)
        return torch.stack([step_norm, loss_norm], dim=-1)

    def forward(self, loss_val, step):
        state_in = self._state_in(loss_val, step)
        new_h, d_mu, d_ls = [], [], []
        for i in range(self.action_dim):
            h_prev = self.hidden[i]
            h_i = self.state_nets[i](state_in, h_prev) if h_prev is not None else self.state_nets[i](state_in)
            new_h.append(h_i)
            mu_i, ls_i = self.out_nets[i](h_i).split(1, dim=-1)
            d_mu.append(mu_i)
            d_ls.append(ls_i)
        self.hidden = [h.detach() for h in new_h]
        mu = self.mu_base + torch.cat(d_mu, dim=1).squeeze(0)
        log_sigma = (self.log_sigma_base + torch.cat(d_ls, dim=1).squeeze(0)).clamp(-6.0, 2.0)
        return mu, log_sigma

    def sample(self, loss_val, step, lo, hi):
        mu, log_sigma = self.forward(loss_val, step)
        sigma = log_sigma.exp()
        eps = torch.randn_like(mu)
        raw = (mu + eps * sigma).detach()
        action = raw.clamp(lo.to(raw.device), hi.to(raw.device))
        log_prob = (-0.5 * ((raw - mu) / sigma) ** 2 - log_sigma - 0.5 * math.log(2 * math.pi)).sum()
        entropy = (log_sigma + 0.5 * (1.0 + math.log(2 * math.pi))).sum()
        return action, log_prob, entropy


def _clone_tensor(x):
    y = x.detach().clone()
    if isinstance(x, nn.Parameter):
        return nn.Parameter(y)
    if x.requires_grad:
        y.requires_grad_(True)
    return y


def _clone_capture(cap):
    out = []
    for x in cap:
        if torch.is_tensor(x):
            out.append(_clone_tensor(x))
        elif isinstance(x, dict):
            out.append(copy.deepcopy(x))
        else:
            out.append(x)
    return tuple(out)


class RLGSController:
    def __init__(self, device="cuda", max_steps=30000,
                 init_log_std=-1.5, densi_init_log_std=-2.0):
        _check_rlgs_torch_version()
        self.device = device
        self.pi_lr = KeywiseGRUPolicy(5, init_log_std=init_log_std, max_steps=max_steps).to(device)
        self.pi_ds = KeywiseGRUPolicy(2, init_log_std=densi_init_log_std, max_steps=max_steps).to(device)
        self.opt_lr = torch.optim.Adam(self.pi_lr.parameters(), lr=1e-4)
        self.opt_ds = torch.optim.Adam(self.pi_ds.parameters(), lr=1e-4)
        self.lo_lr, self.hi_lr = LR_LO.to(device), LR_HI.to(device)
        self.lo_ds, self.hi_ds = DS_LO.to(device), DS_HI.to(device)
        self.lr_scale = torch.ones(5, device=device)
        self.ds_scale = torch.ones(2, device=device)
        self.K = 20
        self.n_lr = 2
        self.n_ds = 4
        self.shuffle = 1000
        self.hidden_interval = 1000
        self.last_loss = 0.2
        self.reward_idx = []
        self.base_extent = None
        self.base_densify = None
        self.base_lrs = None
        self.phase_cams = []
        self.phase_i = 0
        # Keep DS look-ahead views until their real training steps consume them.
        self.future_cams = {}
        self.reward_refresh_pending = False
        self.beta_fn = get_expon_lr_func(lr_init=0.01, lr_final=0.0011, max_steps=max_steps)

    def maybe_reset_hidden(self, iteration):
        if iteration == 0 or iteration % self.hidden_interval != 0:
            return
        self.pi_lr.reset()
        if iteration > 1100:
            self.pi_ds.reset()

    def _ensure_base(self, method):
        if self.base_densify is None:
            self.base_densify = float(method.opt.densify_grad_threshold)
            self.base_extent = float(method.scene_1.cameras_extent)
        if self.base_lrs is None:
            opt = method.opt
            self.base_lrs = {
                "scaling": float(opt.scaling_lr),
                "rotation": float(opt.rotation_lr),
                "opacity": float(opt.opacity_lr),
                "f_dc": float(opt.feature_lr),
            }

    def apply_lr(self, method):
        self._ensure_base(method)
        for group in method.gaussians_1.optimizer.param_groups:
            name = group.get("name", "")
            if name not in LR_KEYS:
                continue
            scale = float(self.lr_scale[LR_KEYS.index(name)])
            if name == "xyz":
                group["lr"] *= scale
            else:
                group["lr"] = self.base_lrs[name] * scale

    def apply_ds(self, method):
        self._ensure_base(method)
        method.opt.densify_grad_threshold = self.base_densify * float(self.ds_scale[0])
        method.scene_1.cameras_extent = self.base_extent * float(self.ds_scale[1])

    def snapshot(self, method):
        masks = None
        mask_opt = None
        mask_grads = None
        if getattr(method, "learnable_mask_logits", None) is not None:
            masks = {k: v.detach().clone() for k, v in method.learnable_mask_logits.items()}
            mask_opt = copy.deepcopy(method.learnable_mask_optimizer.state_dict())
            mask_grads = {k: v.grad.detach().clone() if v.grad is not None else None
                          for k, v in method.learnable_mask_logits.items()}
        g = method.gaussians_1
        appearance = None
        if getattr(method, "use_color_transform", False):
            appearance = {
                "model": copy.deepcopy(method.appearance_transform.state_dict()),
                "optimizer": copy.deepcopy(method.appearance_transform_optimizer.state_dict()),
                "grads": {k: p.grad.detach().clone() if p.grad is not None else None
                          for k, p in method.appearance_transform.named_parameters()},
                "global": method.global_encoding_1.detach().clone(),
                "global_grad": (method.global_encoding_1.grad.detach().clone()
                                if method.global_encoding_1.grad is not None else None),
                "global_optimizer": copy.deepcopy(method.global_encoding_optimizer_1.state_dict()),
            }
        return {
            "gs": _clone_capture(method.gaussians_1.capture()),
            "ema": _clone_capture(method.gaussians_1.capture_ema()),
            "filter_3D": _clone_tensor(method.gaussians_1.filter_3D) if getattr(method.gaussians_1, "filter_3D", None) is not None else None,
            "ds_th": method.opt.densify_grad_threshold,
            "extent": method.scene_1.cameras_extent,
            "step": method.step,
            "sh": method.gaussians_1.active_sh_degree,
            "masks": masks,
            "mask_opt": mask_opt,
            "mask_grads": mask_grads,
            # Author checkpoint capture omits these live densification buffers.
            "densification_stats": {
                name: _clone_tensor(getattr(g, name))
                for name in ("xyz_gradient_accum_abs", "xyz_gradient_accum_abs_max")
            },
            # DS snapshots occur after backward, before the current optimizer step.
            "gaussian_grads": {
                name: getattr(g, name).grad.detach().clone()
                if getattr(g, name).grad is not None else None
                for name in GAUSSIAN_PARAMS
            },
            "appearance": appearance,
        }

    def restore(self, method, snap):
        method.gaussians_1.restore(_clone_capture(snap["gs"]), method.opt)
        ema = _clone_capture(snap["ema"])
        g = method.gaussians_1
        g._xyz_ema = ema[1]
        g._features_dc_ema = ema[2]
        g._features_rest_ema = ema[3]
        g._scaling_ema = ema[4]
        g._rotation_ema = ema[5]
        g._opacity_ema = ema[6]
        if len(ema) > 7:
            g._local_encoding_ema = ema[7]
        if snap.get("filter_3D") is not None:
            g.filter_3D = _clone_tensor(snap["filter_3D"])
        elif len(ema) > 8:
            g.filter_3D = ema[8]
        g.active_sh_degree = snap.get("sh", g.active_sh_degree)
        for name, value in snap["densification_stats"].items():
            setattr(g, name, _clone_tensor(value))
        for name, grad in snap["gaussian_grads"].items():
            getattr(g, name).grad = grad.detach().clone() if grad is not None else None
        method.opt.densify_grad_threshold = snap["ds_th"]
        method.scene_1.cameras_extent = snap["extent"]
        method.step = snap["step"]
        if snap.get("masks") is not None:
            for k, v in snap["masks"].items():
                method.learnable_mask_logits[k].data.copy_(v)
                grad = snap["mask_grads"][k]
                method.learnable_mask_logits[k].grad = grad.clone() if grad is not None else None
            method.learnable_mask_optimizer.load_state_dict(copy.deepcopy(snap["mask_opt"]))
        if snap["appearance"] is not None:
            appearance = snap["appearance"]
            method.appearance_transform.load_state_dict(appearance["model"])
            method.appearance_transform_optimizer.load_state_dict(copy.deepcopy(appearance["optimizer"]))
            for name, param in method.appearance_transform.named_parameters():
                grad = appearance["grads"][name]
                param.grad = grad.clone() if grad is not None else None
            method.global_encoding_1.data.copy_(appearance["global"])
            grad = appearance["global_grad"]
            method.global_encoding_1.grad = grad.clone() if grad is not None else None
            method.global_encoding_optimizer_1.load_state_dict(copy.deepcopy(appearance["global_optimizer"]))

    @torch.no_grad()
    def _psnr(self, method, cams):
        bg = method.background
        vals = []
        for cam in cams:
            gt = cam.original_image.cuda()
            img = render(cam, method.gaussians_1, method.pipe, bg, kernel_size=method.dataset.kernel_size)["render"]
            mse = torch.mean((img - gt) ** 2).clamp_min(1e-10)
            vals.append(10 * torch.log10(1 / mse))
        return torch.stack(vals).mean()

    def _pick_reward_cams(self, method, iteration):
        cams = method.trainCameras_1
        if not cams:
            return []
        if iteration % self.shuffle == 1:
            self.reward_refresh_pending = True
        # Retain the DS trial's hold-out split while its future views replay.
        if (not self.reward_idx) or (
            self.reward_refresh_pending and not self.future_cams and iteration % self.K == 1
        ):
            idx = list(range(len(cams)))
            random.shuffle(idx)
            self.reward_idx = idx[:2] if len(idx) >= 2 else idx
            self.reward_refresh_pending = False
        return [cams[i] for i in self.reward_idx]

    def held_out_names(self, method):
        if not self.reward_idx or not method.trainCameras_1:
            return set()
        return {method.trainCameras_1[i].image_name for i in self.reward_idx}

    def reserve_phase_views(self, method, iteration):
        hold = self.held_out_names(method)
        self.phase_cams = [
            self.future_cams[iteration + k] if iteration + k in self.future_cams
            else method.draw_train_cam(exclude=hold)
            for k in range(self.K)
        ]
        self.phase_i = 0

    def rewind_phase(self):
        self.phase_i = 0

    def next_phase_cam(self, iteration=None, simulate=False):
        cam = None
        if self.phase_i < len(self.phase_cams):
            cam = self.phase_cams[self.phase_i]
            self.phase_i += 1
        if iteration in self.future_cams:
            cam = self.future_cams[iteration]
            if not simulate:
                del self.future_cams[iteration]
        return cam

    def run_phase(self, method, iteration):
        reward_cams = self._pick_reward_cams(method, iteration)
        if len(reward_cams) < 1:
            return
        self.reserve_phase_views(method, iteration)
        self._ensure_base(method)
        beta = float(self.beta_fn(iteration))
        snap = self.snapshot(method)
        with torch.no_grad():
            base_psnr = self._psnr(method, reward_cams)

        best_r, best_a = -1e9, self.lr_scale.detach().clone()
        for _ in range(self.n_lr):
            self.restore(method, snap)
            self.rewind_phase()
            action, logp, ent = self.pi_lr.sample(self.last_loss, iteration, self.lo_lr, self.hi_lr)
            self.lr_scale = action.detach()
            for k in range(self.K):
                # Compare LR candidates without stochastic split/clone/prune.
                method._train_step_core(iteration + k, simulate=True,
                                        allow_densification=False)
            r = (self._psnr(method, reward_cams) - base_psnr).detach()
            loss = -(logp * r) - beta * ent
            self.opt_lr.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.pi_lr.parameters(), 2.4)
            self.opt_lr.step()
            if float(r) > best_r:
                best_r, best_a = float(r), action.detach().clone()
        self.restore(method, snap)
        self.lr_scale = best_a
        print(f"[RLGS] iter={iteration} best_lr={[round(x, 3) for x in best_a.tolist()]} R={best_r:.4f} β={beta:.4f}")

        self.rewind_phase()

    def should_densify(self, method, iteration):
        return (method.opt.densify_from_iter < iteration < method.opt.densify_until_iter
                and iteration % method.opt.densification_interval == 0)

    def run_densification(self, method, iteration, finish_step):
        """Select topology at the real densification point, then keep that topology.

        The caller has already accumulated this step's gradients. Each branch
        densifies, finishes the pending optimizer/EMA/mask updates, and trains
        K future steps. Only the pre-rollout, post-densify snapshot is committed;
        the caller then finishes the real current step exactly once.
        """
        if not self.should_densify(method, iteration):
            return
        self._ensure_base(method)
        reward_cams = self._pick_reward_cams(method, iteration)
        if not reward_cams:
            with torch.no_grad():
                method._densify_now(iteration)
            return
        snap = self.snapshot(method)
        old_scale = self.ds_scale.detach().clone()
        old_phase, old_phase_i = self.phase_cams, self.phase_i
        hold = self.held_out_names(method)
        future_steps = min(self.K, method.opt.iterations - iteration)
        remaining_phase = old_phase[old_phase_i:]
        future_cams = []
        for k in range(future_steps):
            future_iteration = iteration + k + 1
            if future_iteration in self.future_cams:
                cam = self.future_cams[future_iteration]
            elif k < len(remaining_phase):
                cam = remaining_phase[k]
            else:
                cam = method.draw_train_cam(exclude=hold)
            future_cams.append(cam)
        self.phase_cams = future_cams

        def start_candidate(action):
            self.restore(method, snap)
            self.ds_scale = action.detach().clone()
            self.apply_ds(method)
            self.rewind_phase()
            with torch.no_grad():
                method._densify_now(iteration)

        def score_candidate():
            # Match the remainder of the current real training step before t+1.
            with torch.no_grad():
                finish_step()
            for k in range(1, future_steps + 1):
                method._train_step_core(iteration + k, simulate=True)
            return self._psnr(method, reward_cams)

        try:
            # TMGS compares DS candidates against a default-densify rollout.
            start_candidate(torch.ones_like(self.ds_scale))
            default_psnr = score_candidate()
            best_r, best_a, best_state = -math.inf, None, None
            for _ in range(self.n_ds):
                action, logp, ent = self.pi_ds.sample(self.last_loss, iteration, self.lo_ds, self.hi_ds)
                start_candidate(action)
                candidate_state = self.snapshot(method)
                r = (score_candidate() - default_psnr).detach()
                if not torch.isfinite(r):
                    raise RuntimeError("Non-finite RLGS densification reward")
                loss = -(logp * r) - 0.01 * ent
                self.opt_ds.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.pi_ds.parameters(), 2.4)
                self.opt_ds.step()
                if float(r) > best_r:
                    best_r, best_a = float(r), action.detach().clone()
                    best_state = candidate_state
            if best_state is None:
                raise RuntimeError("RLGS densification needs at least one candidate")
            self.restore(method, best_state)
            self.ds_scale = best_a
            self.apply_ds(method)
            print(f"[RLGS] iter={iteration} best_ds={[round(x, 3) for x in best_a.tolist()]} R={best_r:.4f}")
            self.future_cams.update({iteration + k + 1: cam for k, cam in enumerate(future_cams)})
        except Exception:
            self.restore(method, snap)
            self.ds_scale = old_scale
            raise
        finally:
            self.phase_cams, self.phase_i = old_phase, old_phase_i
