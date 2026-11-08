import math
import torch
import torch.nn as nn

class ScalarPolicyWithSchedStateGRUKeywiseCrossKey(nn.Module):
    """
    每个 key 使用独立的 GRUCell，并在它们之间引入跨-Key 自注意力通信，
    接口和输出与原类保持一致。
    """
    def __init__(
        self,
        key_info_list,
        init_std: float = 0.0,
        hidden: int = 32,
        clamp: bool = True,
        detach: bool = True,
        max_steps: int = 30_000,
        ema_scale: float = 5.0,
        num_heads: int = 4,
    ):
        super().__init__()
        self.keys         = [name for name, *_ in key_info_list]
        self.num_keys     = len(self.keys)
        self.hidden_size  = hidden


        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.register_buffer('mins', torch.as_tensor(mins))    # [K]
        self.register_buffer('maxs', torch.as_tensor(maxs))    # [K]


        self.base_mu      = nn.Parameter(torch.ones(self.num_keys))
        self.base_log_std = nn.Parameter(torch.ones(self.num_keys) * init_std)


        self.state_nets = nn.ModuleList([
            nn.GRUCell(2, hidden) for _ in range(self.num_keys)
        ])
        self.out_nets   = nn.ModuleList([
            nn.Linear(hidden, 2) for _ in range(self.num_keys)
        ])


        self.cross_key_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            batch_first=True
        )
        self.cross_key_norm = nn.LayerNorm(hidden)


        self.clamp      = clamp
        self.detach     = detach
        self.max_steps  = float(max_steps)
        self.ema_scale  = float(ema_scale)


        for cell in self.state_nets:
            for name, param in cell.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        for out in self.out_nets:
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)


        self.hidden = [None] * self.num_keys

    def forward(
        self,
        *,
        step:   torch.Tensor | int | None,
        loss:   torch.Tensor | float | None,
        batch_size: int = 1,
        device=None,
    ):
        if device is None:
            device = self.base_mu.device

        # --------- scalar -> Tensor[B]
        step_t = torch.as_tensor(step,  device=device, dtype=torch.float32).view(-1) if step is not None else torch.zeros(batch_size, device=device)
        loss_t = torch.as_tensor(loss,  device=device, dtype=torch.float32).view(-1) if loss is not None else torch.zeros(batch_size, device=device)
        if step_t.numel() == 1:
            step_t = step_t.expand(batch_size)
        if loss_t.numel() == 1:
            loss_t = loss_t.expand(batch_size)


        step_norm = (step_t / self.max_steps).clamp_(0., 1.)
        loss_norm = torch.tanh(torch.log1p(loss_t) * self.ema_scale)
        state_in = torch.stack([step_norm, loss_norm], dim=-1)   # [B,2]


        new_h = []
        for i in range(self.num_keys):
            cell   = self.state_nets[i]
            h_prev = self.hidden[i]

            h_i = cell(state_in, h_prev) if h_prev is not None else cell(state_in)
            new_h.append(h_i)


        # stack -> [B, K, H]
        h_stack = torch.stack(new_h, dim=1)

        h_comm, _ = self.cross_key_attn(h_stack, h_stack, h_stack)

        h_stack = self.cross_key_norm(h_stack + h_comm)

        new_h = [h_stack[:, i, :] for i in range(self.num_keys)]


        self.hidden = [h.detach() for h in new_h]


        d_mu_list = []
        d_ls_list = []
        for i in range(self.num_keys):
            delta_i = self.out_nets[i](new_h[i])   # [B,2]
            mu_i, ls_i = delta_i.chunk(2, dim=-1) if False else delta_i.split(1, dim=-1)
            d_mu_list.append(mu_i)
            d_ls_list.append(ls_i)
        d_mu = torch.cat(d_mu_list, dim=1)   # [B,K]
        d_ls = torch.cat(d_ls_list, dim=1)


        mu      = self.base_mu + d_mu
        log_std = self.base_log_std + d_ls
        if self.clamp:
            log_std = torch.clamp(log_std, -6.0, 2.0)
        std = torch.exp(log_std)


        eps = torch.randn(batch_size, self.num_keys, device=device)
        raw = mu + eps * std
        if self.detach:
            raw = raw.detach()


        action = torch.clamp(raw,
                             self.mins.to(device).unsqueeze(0),
                             self.maxs.to(device).unsqueeze(0))

        # --------- log_prob & entropy
        logp = -0.5 * (((raw - mu) / std) ** 2) - log_std - 0.5 * math.log(2 * math.pi)
        entropy = log_std + 0.5 * (1. + math.log(2 * math.pi))


        return {
            k: {
                "action":   action[:, i],
                "log_prob": logp  [:, i],
                "entropy":  entropy[:, i],
            }
            for i, k in enumerate(self.keys)
        }




class ScalarPolicyWithSchedStateLSTMKeywise(nn.Module):
    """
    每个 key 使用独立的 LSTMCell，接口和输出与原 GRU 版保持一致。
    """
    def __init__(
        self,
        key_info_list,
        init_std: float = 0.0,
        hidden: int = 32,
        clamp: bool = True,
        detach: bool = True,
        max_steps: int = 30_000,
        ema_scale: float = 5.0,
    ):
        super().__init__()
        self.keys        = [name for name, *_ in key_info_list]
        self.num_keys    = len(self.keys)
        self.hidden_size = hidden


        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.register_buffer('mins', torch.as_tensor(mins))   # [K]
        self.register_buffer('maxs', torch.as_tensor(maxs))   # [K]


        self.base_mu      = nn.Parameter(torch.ones(self.num_keys))
        self.base_log_std = nn.Parameter(torch.ones(self.num_keys) * init_std)


        self.state_nets = nn.ModuleList([
            nn.LSTMCell(2, hidden) for _ in range(self.num_keys)
        ])
        self.out_nets   = nn.ModuleList([
            nn.Linear(hidden, 2) for _ in range(self.num_keys)
        ])


        self.clamp     = clamp
        self.detach    = detach
        self.max_steps = float(max_steps)
        self.ema_scale = float(ema_scale)


        for cell in self.state_nets:
            for name, param in cell.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        for out in self.out_nets:
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)


        self.hidden = [(None, None) for _ in range(self.num_keys)]

    def forward(
        self,
        *,
        step:        torch.Tensor | int | None,
        loss:        torch.Tensor | float | None,
        batch_size:  int = 1,
        device=None,
    ):
        if device is None:
            device = self.base_mu.device

        # scalar -> Tensor[B]
        step_t = torch.as_tensor(step,  device=device, dtype=torch.float32).view(-1) if step is not None else torch.zeros(batch_size, device=device)
        loss_t = torch.as_tensor(loss,  device=device, dtype=torch.float32).view(-1) if loss is not None else torch.zeros(batch_size, device=device)
        if step_t.numel() == 1:
            step_t = step_t.expand(batch_size)
        if loss_t.numel() == 1:
            loss_t = loss_t.expand(batch_size)


        step_norm = (step_t / self.max_steps).clamp_(0., 1.)
        loss_norm = torch.tanh(torch.log1p(loss_t) * self.ema_scale)
        state_in  = torch.stack([step_norm, loss_norm], dim=-1)   # [B,2]


        new_states = []  # will hold (h_i, c_i)
        h_outputs  = []  # list of h_i for each key
        for i in range(self.num_keys):
            cell   = self.state_nets[i]
            h_prev, c_prev = self.hidden[i]
            if h_prev is None or c_prev is None:

                h_prev = torch.zeros(batch_size, self.hidden_size, device=device)
                c_prev = torch.zeros(batch_size, self.hidden_size, device=device)
            h_i, c_i = cell(state_in, (h_prev, c_prev))  # [B,hidden]
            new_states.append((h_i, c_i))
            h_outputs.append(h_i)


        self.hidden = [(h_i.detach(), c_i.detach()) for (h_i, c_i) in new_states]


        d_mu_list = []
        d_ls_list = []
        for i in range(self.num_keys):
            delta_i = self.out_nets[i](h_outputs[i])  # [B,2]
            mu_i, ls_i = delta_i.split(1, dim=-1)
            d_mu_list.append(mu_i)
            d_ls_list.append(ls_i)
        d_mu = torch.cat(d_mu_list, dim=1)  # [B,K]
        d_ls = torch.cat(d_ls_list, dim=1)


        mu      = self.base_mu + d_mu
        log_std = self.base_log_std + d_ls
        if self.clamp:
            log_std = torch.clamp(log_std, -6.0, 2.0)
        std = torch.exp(log_std)


        eps = torch.randn(batch_size, self.num_keys, device=device)
        raw = mu + eps * std
        if self.detach:
            raw = raw.detach()


        action = torch.clamp(raw,
                             self.mins.to(device).unsqueeze(0),
                             self.maxs.to(device).unsqueeze(0))

        # log_prob & entropy
        logp = -0.5 * (((raw - mu) / std) ** 2) - log_std - 0.5 * math.log(2 * math.pi)
        entropy = log_std + 0.5 * (1. + math.log(2 * math.pi))


        return {
            k: {
                "action":   action[:, i],
                "log_prob": logp  [:, i],
                "entropy":  entropy[:, i],
            }
            for i, k in enumerate(self.keys)
        }





class SpatialAxisPolicyWithSchedGRUKeywise(nn.Module):
    """
    用 (step, loss) -> 产生三个轴上的标量:
        a_x: [Nx]
        a_y: [Ny]
        a_z: [Nz]
    再由外部组合成 w(x,y,z) = a_x[i] * a_y[j] * a_z[k]

    这相当于把原来的 key-wise policy 改成了 "axis-wise" policy。
    
    仍然吃 key_info_list

    每个 key（xyz / scaling / rotation / opacity / f_dc）
    → 各自一套 (ax, ay, az)

    轴分解：Nx + Ny + Nz，而不是 Nx * Ny * Nz

    输出可直接用于 per-Gaussian / per-parameter gradient scaling

    """
    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        init_std: float = 0.0,
        hidden: int = 32,
        clamp_log_std: bool = True,
        detach_action: bool = True,
        max_steps: int = 30_000,
        ema_scale: float = 5.0,
        init_mu: float = 0.0,

        min_val: float = 0.1,
        max_val: float = 2.0,
    ):
        super().__init__()
        self.nx, self.ny, self.nz = nx, ny, nz
        self.num_keys = nx + ny + nz
        self.hidden_size = hidden


        mins = torch.full((self.num_keys,), float(min_val))
        maxs = torch.full((self.num_keys,), float(max_val))
        self.register_buffer("mins", mins)    # [K]
        self.register_buffer("maxs", maxs)    # [K]


        self.base_mu      = nn.Parameter(init_mu * torch.ones(self.num_keys))
        self.base_log_std = nn.Parameter(torch.ones(self.num_keys) * init_std)


        self.state_nets = nn.ModuleList([
            nn.GRUCell(2, hidden) for _ in range(self.num_keys)
        ])
        self.out_nets   = nn.ModuleList([
            nn.Linear(hidden, 2) for _ in range(self.num_keys)
        ])

        self.clamp_log_std = clamp_log_std
        self.detach_action = detach_action
        self.max_steps     = float(max_steps)
        self.ema_scale     = float(ema_scale)

        # init
        for cell in self.state_nets:
            for name, param in cell.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        for out in self.out_nets:
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)


        self.hidden = None


    def set_hidden_state(self, batch_size: int = 1, device: str | torch.device = "cuda"):
        self.hidden = [
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(self.num_keys)
        ]

    def forward(
        self,
        *,
        step:   torch.Tensor | int | None,
        loss:   torch.Tensor | float | None,
        batch_size: int = 1,
        device=None,
    ):
        if device is None:
            device = self.base_mu.device

        # scalar → Tensor[B]
        step_t = (torch.as_tensor(step, device=device, dtype=torch.float32)
                  if step is not None else torch.zeros((), device=device))
        loss_t = (torch.as_tensor(loss, device=device, dtype=torch.float32)
                  if loss is not None else torch.zeros((), device=device))

        step_t = step_t.view(-1)
        loss_t = loss_t.view(-1)
        if step_t.numel() == 1:
            step_t = step_t.expand(batch_size)
        if loss_t.numel() == 1:
            loss_t = loss_t.expand(batch_size)


        step_norm = (step_t / self.max_steps).clamp_(0., 1.)      # [B]
        loss_norm = torch.tanh(torch.log1p(loss_t) * self.ema_scale)  # [B]
        state_in  = torch.stack([step_norm, loss_norm], dim=-1)   # [B,2]

        if self.hidden is None or len(self.hidden) != self.num_keys:

            self.set_hidden_state(batch_size=batch_size, device=device)

        new_h = []
        d_mu  = []
        d_ls  = []
        for i in range(self.num_keys):
            cell   = self.state_nets[i]
            h_prev = self.hidden[i]
            h_i    = cell(state_in, h_prev) if h_prev is not None else cell(state_in)
            new_h.append(h_i)

            delta_i = self.out_nets[i](h_i)  # [B,2]
            mu_i, ls_i = delta_i.split(1, dim=-1)
            d_mu.append(mu_i)
            d_ls.append(ls_i)


        self.hidden = [h.detach() for h in new_h]


        d_mu  = torch.cat(d_mu, dim=1)
        d_ls  = torch.cat(d_ls, dim=1)


        mu      = self.base_mu + d_mu
        log_std = self.base_log_std + d_ls
        if self.clamp_log_std:
            log_std = torch.clamp(log_std, -6.0, 2.0)
        std = torch.exp(log_std)


        eps = torch.randn(batch_size, self.num_keys, device=device)
        raw = mu + eps * std
        if self.detach_action:
            raw = raw.detach()


        mins = self.mins.to(device).unsqueeze(0)   # [1,K]
        maxs = self.maxs.to(device).unsqueeze(0)   # [1,K]
        action = torch.clamp(raw, mins, maxs)      # [B,K]


        logp = -0.5 * (((raw - mu) / std) ** 2) - log_std - 0.5 * math.log(2 * math.pi)
        entropy = log_std + 0.5 * (1. + math.log(2 * math.pi))


        ax = action[:, :self.nx]                         # [B, Nx]
        ay = action[:, self.nx:self.nx + self.ny]        # [B, Ny]
        az = action[:, self.nx + self.ny:]               # [B, Nz]

        logp_x = logp[:, :self.nx]
        logp_y = logp[:, self.nx:self.nx + self.ny]
        logp_z = logp[:, self.nx + self.ny:]

        ent_x = entropy[:, :self.nx]
        ent_y = entropy[:, self.nx:self.nx + self.ny]
        ent_z = entropy[:, self.nx + self.ny:]

        return {
            "ax": {
                "action":   ax,      # [B, Nx]
                "log_prob": logp_x,  # [B, Nx]
                "entropy":  ent_x,   # [B, Nx]
            },
            "ay": {
                "action":   ay,      # [B, Ny]
                "log_prob": logp_y,
                "entropy":  ent_y,
            },
            "az": {
                "action":   az,      # [B, Nz]
                "log_prob": logp_z,
                "entropy":  ent_z,
            },

        }

    def build_weight_grid(self, ax: torch.Tensor, ay: torch.Tensor, az: torch.Tensor):
        """
        把 [Nx], [Ny], [Nz] 组合成 [Nx, Ny, Nz] 的 weight grid:
            w[i,j,k] = ax[i] * ay[j] * az[k]
        假定 batch_size = 1，如果有 batch 你可以自己再扩。
        """
        # ax: [1,Nx] or [Nx]
        if ax.dim() == 2:
            ax = ax[0]
            ay = ay[0]
            az = az[0]

        # [Nx,1,1] * [1,Ny,1] * [1,1,Nz] → [Nx,Ny,Nz]
        wx = ax.view(self.nx, 1, 1)
        wy = ay.view(1, self.ny, 1)
        wz = az.view(1, 1, self.nz)
        w_grid = wx * wy * wz
        return w_grid