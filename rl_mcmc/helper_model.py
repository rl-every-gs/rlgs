import math 

#import tinycudann as tcnn
import json 
        
import torch 
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=2):
        """
        d_model: embedding dimension for transformer.
        nhead: number of attention heads.
        num_layers: number of transformer encoder layers.
        """
        super(Transformer, self).__init__()
        # Embed input Nx6 into d_model dimensions.
        self.embedding = nn.Linear(10, d_model)
        
        # Build transformer encoder layers.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final projection to 48 dimensions: 6 attributes * 8 parameters each.
        self.fc = nn.Linear(d_model, 48)
        
        # Learnable log standard deviation (for Gaussian sampling), shape (48,).
        self.log_std = nn.Parameter(torch.zeros(48))
    
    def forward(self, prior_xyz, prior_scaling, prior_rotation):
        """
        prior_xyz: Tensor of shape (N, 3) representing the point positions.
        prior_scaling: Tensor of shape (N, 3) representing the point scales.
        prior_rotation: (Unused in this implementation, reserved for future use.)
        
        Returns a dictionary with keys corresponding to attributes:
          ["_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest"]
        Each value is a dict with:
          - "action": the sampled learning rate multipliers (Tensor of shape (N,8))
          - "log_prob": the log probability of the sampled actions (Tensor of shape (N,8))
        """
        # Concatenate the two inputs to form an Nx6 tensor.
        # (Assumes prior_xyz and prior_scaling are both of shape (N,3))
        x = torch.cat([prior_xyz, prior_scaling, prior_rotation], dim=-1)  # shape: (N, 6)
        
        # Apply linear embedding.
        x = self.embedding(x)  # shape: (N, d_model)
        
        # Transformer encoder expects input shape (seq_len, batch_size, d_model).
        # Here we treat each point as a token (seq_len = N) and use batch_size=1.
        x = x.unsqueeze(1)  # shape: (N, 1, d_model)
        x = self.transformer_encoder(x)  # shape: (N, 1, d_model)
        x = x.squeeze(1)  # shape: (N, d_model)
        
        # Project to 48 dimensions.
        means = self.fc(x)  # shape: (N, 48)
        
        # Expand the learnable log_std to match the shape.
        std = torch.exp(self.log_std)         # shape: (48,)
        std = std.unsqueeze(0).expand_as(means) # shape: (N, 48)
        
        # Sample from a Gaussian using the reparameterization trick.
        eps = torch.randn_like(means)
        actions = means + std * eps
        
        # Compute log probabilities under the Gaussian.
        #log_prob = -0.5 * (((action - mean)/std)^2) - log(std) - 0.5*log(2*pi)
        log_probs = -0.5 * (((actions - means) / std) ** 2) - self.log_std - 0.5 * math.log(2 * math.pi)




        # Split the 48-dimensional outputs into 6 groups (each of size 8).
        actions_split = torch.split(actions, 8, dim=-1)
        log_probs_split = torch.split(log_probs, 8, dim=-1)
        
        keys = ["_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest"]
        lr_adjust_dict = {
            k: {"action": a, "log_prob": lp} 
            for k, a, lp in zip(keys, actions_split, log_probs_split)
        }
        
        return lr_adjust_dict
    
    
    


class PETransformer(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=2, max_len=5000):
        super(PETransformer, self).__init__()
        self.embedding = nn.Linear(6, d_model)
        
        # Positional encoding module.
        self.positional_encoding = self._generate_positional_encoding(max_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, 48)
        self.log_std = nn.Parameter(torch.zeros(48))
    
    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        return pe  # This is a buffer you may register as a persistent buffer in a full implementation.
    
    def forward(self, prior_xyz, prior_scaling, prior_rotation):
        # Concatenate spatial and scale information.
        x = torch.cat([prior_xyz, prior_scaling], dim=-1)  # shape: (N, 6)
        x = self.embedding(x)  # shape: (N, d_model)
        
        # Add positional encoding if the number of points (N) is less than max_len.
        pos_enc = self.positional_encoding[:x.size(0), :, :].squeeze(1)  # shape: (N, d_model)
        x = x + pos_enc
        
        # Transformer encoder expects input shape (seq_len, batch_size, d_model).
        x = x.unsqueeze(1)  # shape: (N, 1, d_model)
        x = self.transformer_encoder(x)  # shape: (N, 1, d_model)
        x = x.squeeze(1)  # shape: (N, d_model)
        
        means = self.fc(x)  # shape: (N, 48)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(means)  # shape: (N, 48)
        eps = torch.randn_like(means)
        actions = means + std * eps
        log_probs = -0.5 * (((actions - means) / std) ** 2) - self.log_std - 0.5 * math.log(2 * math.pi)
        
        actions_split = torch.split(actions, 8, dim=-1)
        log_probs_split = torch.split(log_probs, 8, dim=-1)
        
        keys = ["_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest"]
        lr_adjust_dict = {
            k: {"action": a, "log_prob": lp} 
            for k, a, lp in zip(keys, actions_split, log_probs_split)
        }
        
        return lr_adjust_dict
    



class Critic(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        """
        A simple MLP-based critic that takes a state representation (here, concatenated [xyz, scaling])
        and outputs a scalar value estimate.
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    
    


import json
import math
import torch
import torch.nn as nn
#import tinycudann as tcnn   


# class PointPolicy(nn.Module):
#     def __init__(self, attr_dim=59, xyz_dim=3, config_path="data.json"):
#         super().__init__()
#         # Channel splits for output
#         self.channels = [3, 3, 4, 1, 3, 45]
#         n_output_dims = sum(self.channels)  # 59
        
#         # Load Tiny-cuda-nn network config
#         with open(config_path) as f:
#             config = json.load(f)
        
#         # Total input dims = xyz + attribute_grad
#         self.n_input_dims = xyz_dim + attr_dim
        
#         # HashGrid encoding config
#         encoding = {
#             "otype": "HashGrid",
#             "n_levels": 16,
#             "n_features_per_level": 2,
#             "log2_hashmap_size": 19,
#             "base_resolution": 16,
#             "per_level_scale": 1.5,
#             "n_input_dims": self.n_input_dims
#         }

#         # Build encoding + MLP
#         self.encoding = tcnn.Encoding(self.n_input_dims, encoding)
#         self.network  = tcnn.Network(self.encoding.n_output_dims,
#                                      n_output_dims,
#                                      config["network"])
#         self.policy_network = nn.Sequential(self.encoding, self.network)
        
#         # Learnable log-std for Gaussian policy
#         self.log_std = nn.Parameter(torch.zeros(n_output_dims))

#     def forward(self, xyz: torch.Tensor, grads: torch.Tensor):
#         """
#         Args:
#           xyz:   [N, 3]  coordinates in [-1,1] (or pre-normalized to [0,1])
#           grads: [N, attr_dim]  gradient features normalized to [0,1]
#         Returns:
#           dict mapping each channel key to {"action": Tensor, "log_prob": Tensor}
#         """
#         # 1) concatenate inputs
#         x = torch.cat([xyz, grads], dim=-1)  # [N, n_input_dims]
#         # 2) forward through encoding + network
#         means = self.policy_network(x)       # [N, 59]
#         # 3) build Gaussian distribution
#         log_std = torch.clamp(self.log_std, -20, -10)
#         std = torch.exp(log_std).unsqueeze(0).expand_as(means)  # [N,59]
#         eps = torch.randn_like(means)
#         actions = means + std * eps          # [N,59]
#         # 4) compute per-dim log-prob
#         log_probs = -0.5 * (((actions - means) / std) ** 2) \
#                     - log_std - 0.5 * math.log(2 * math.pi)
        
#         # 5) split outputs by channels
#         actions_split   = torch.split(actions,   self.channels, dim=-1)
#         logp_split      = torch.split(log_probs, self.channels, dim=-1)
#         keys = ["_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest"]
        
#         # 6) assemble dict
#         grad_adjust_dict = {
#             k: {"action": a, "log_prob": lp}
#             for k, a, lp in zip(keys, actions_split, logp_split)
#         }
#         return grad_adjust_dict


from torch.distributions import Normal, Independent

class PointPolicy(nn.Module):
    """
    Split encoding: HashGrid for xyz (3 dims) + MLP for high-dim grads.
    """
    def __init__(self,
                 attr_dim=59,
                 xyz_min=(0.0,0.0,0.0),
                 xyz_max=(1.0,1.0,1.0),
                 grad_min=0.0,
                 grad_max=1.0,
                 config_path="data.json"):
        super().__init__()
        # Channel splits
        self.channels = [3,3,4,1,3,45]
        n_output_dims = sum(self.channels)  # 59
        
        # Load Tiny-cuda-nn network config
        with open(config_path) as f:
            config = json.load(f)
        
        # Normalization buffers
        self.register_buffer('xyz_min', torch.tensor(xyz_min))
        self.register_buffer('xyz_range', (torch.tensor(xyz_max) - torch.tensor(xyz_min)).clamp(min=1e-6))
        if isinstance(grad_min, (float,int)):
            grad_min = torch.full((attr_dim,), float(grad_min))
            grad_max = torch.full((attr_dim,), float(grad_max))
        else:
            grad_min = torch.tensor(grad_min)
            grad_max = torch.tensor(grad_max)
        self.register_buffer('grad_min', grad_min)
        self.register_buffer('grad_range', (grad_max - grad_min).clamp(min=1e-6))
        
        # 1) HashGrid only for xyz (3 dims)
        xyz_enc_cfg = {
            "otype": "HashGrid", "n_levels": 16, "n_features_per_level": 2,
            "log2_hashmap_size": 19, "base_resolution": 16, "per_level_scale": 1.5,
            "n_input_dims": 3
        }
        self.xyz_encoding = tcnn.Encoding(3, xyz_enc_cfg)
        
        # 2) MLP for grads -> reduce to a smaller vector
        self.grad_mlp = nn.Sequential(
            nn.Linear(attr_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        
        # 3) Combined network
        joint_dim = self.xyz_encoding.n_output_dims + 32
        self.network = tcnn.Network(joint_dim, n_output_dims, config["network"])
        
        # Learnable log-std
        #self.log_std = nn.Parameter(torch.zeros(n_output_dims))
        self.log_std = nn.Parameter(torch.ones(n_output_dims)*-1.0) 

    def forward(self, xyz: torch.Tensor, grads: torch.Tensor):
        # normalize inputs
        xyz_norm = ((xyz - self.xyz_min) / self.xyz_range).clamp(0,1)
        grads_norm = ((grads - self.grad_min) / self.grad_range).clamp(0,1)
        
        # encode xyz
        xyz_feat = self.xyz_encoding(xyz_norm)  # [N, xyz_enc_dim]
        # encode grads
        grad_feat = self.grad_mlp(grads_norm)   # [N, 32]
        
        # concatenate
        x = torch.cat([xyz_feat, grad_feat], dim=-1)  # [N, joint_dim]
        
        # network -> means
        means = self.network(x)  # [N, 59]
        
        # distribution
        log_std = torch.clamp(self.log_std, -20, -1)
        std = torch.exp(log_std).unsqueeze(0).expand_as(means)
        eps = torch.randn_like(means)
        actions = means + std * eps
        
        log_probs = -0.5 * (((actions - means) / std)**2) - log_std - 0.5*math.log(2*math.pi)
        
        # split
        acts = torch.split(actions, self.channels, dim=-1)
        lps  = torch.split(log_probs, self.channels, dim=-1)
        keys = ["_xyz","_scaling","_rotation","_opacity","_features_dc","_features_rest"]
        
        return {k: {"action": a, "log_prob": lp} for k,a,lp in zip(keys, acts, lps)}

# class PointPolicy(nn.Module):
#     def __init__(self,
#                  attr_dim=59,
#                  xyz_min=(0.0, 0.0, 0.0),
#                  xyz_max=(1.0, 1.0, 1.0),
#                  grad_min=-10.0,
#                  grad_max=10.0,
#                  config_path="data.json"):
#         super().__init__()
#         # Channel splits for output
#         self.channels = [3, 3, 4, 1, 3, 45]
#         n_output_dims = sum(self.channels)  # 59

#         # Load network config
#         with open(config_path) as f:
#             config = json.load(f)

#         # Save normalization stats as buffers
#         xyz_min = torch.tensor(xyz_min, dtype=torch.float32)
#         xyz_max = torch.tensor(xyz_max, dtype=torch.float32)
#         self.register_buffer('xyz_min', xyz_min)
#         self.register_buffer('xyz_range', (xyz_max - xyz_min).clamp(min=1e-6))

#         if grad_min is None or grad_max is None:
#             # default to [0,1] if not provided
#             grad_min = torch.zeros(attr_dim)
#             grad_max = torch.ones(attr_dim)
#         else:
#             grad_min = torch.tensor(grad_min, dtype=torch.float32)
#             grad_max = torch.tensor(grad_max, dtype=torch.float32)
#         self.register_buffer('grad_min', grad_min)
#         self.register_buffer('grad_range', (grad_max - grad_min).clamp(min=1e-6))

#         # Build Tiny-cuda-nn encoding + network
#         self.n_input_dims = 3 + attr_dim
#         encoding_cfg = {
#             "otype": "HashGrid",
#             "n_levels": 16,
#             "n_features_per_level": 2,
#             "log2_hashmap_size": 19,
#             "base_resolution": 16,
#             "per_level_scale": 1.5,
#             "n_input_dims": self.n_input_dims
#         }
#         self.encoding = tcnn.Encoding(self.n_input_dims, encoding_cfg)
#         self.network = tcnn.Network(self.encoding.n_output_dims,
#                                    n_output_dims,
#                                    config["network"])
#         self.policy_network = nn.Sequential(self.encoding, self.network)

#         # Learnable log-std for Gaussian policy
#         self.log_std = nn.Parameter(torch.zeros(n_output_dims))

#     def forward(self, xyz: torch.Tensor, grads: torch.Tensor):
#         """
#         Args:
#           xyz:   [N, 3] coordinates in original range [xyz_min, xyz_min+xyz_range]
#           grads: [N, attr_dim] in original range [grad_min, grad_min+grad_range]
#         Returns:
#           grad_adjust_dict: dict of channel-wise denormalized actions and log_probs
#         """
#         # Normalize inputs to [0,1]
#         xyz_norm = (xyz - self.xyz_min) / self.xyz_range
#         xyz_norm = xyz_norm.clamp(0.0, 1.0)

#         grads_norm = (grads - self.grad_min) / self.grad_range
#         grads_norm = grads_norm.clamp(0.0, 1.0)

#         # Concatenate
#         inp = torch.cat([xyz_norm, grads_norm], dim=-1)  # [N, n_input_dims]

#         # Forward through policy network to predict normalized means
#         means_norm = self.policy_network(inp)  # [N, 59]

#         # Gaussian sampling in normalized space
#         log_std = torch.clamp(self.log_std, -20, -10)
#         std_norm = torch.exp(log_std).unsqueeze(0).expand_as(means_norm)
#         eps = torch.randn_like(means_norm)
#         actions_norm = means_norm + std_norm * eps  # [N,59]

#         # Compute log-prob (in normalized space)
#         log_probs_norm = -0.5 * (((actions_norm - means_norm) / std_norm) ** 2) \
#                          - log_std - 0.5 * math.log(2 * math.pi)

#         # Denormalize actions back to original grad range
#         actions = actions_norm * self.grad_range.unsqueeze(0) + self.grad_min.unsqueeze(0)

#         # Split by channels
#         actions_split = torch.split(actions, self.channels, dim=-1)
#         logp_split = torch.split(log_probs_norm, self.channels, dim=-1)
#         keys = ["_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest"]

#         grad_adjust_dict = {
#             k: {"action": a, "log_prob": lp}
#             for k, a, lp in zip(keys, actions_split, logp_split)
#         }
#         return grad_adjust_dict


import torch
import torch.nn as nn
import json

class PointPolicyTransformer(nn.Module):
    def __init__(self, n_input_dims=6, d_model=64, nhead=8, num_layers=3, dropout=0.1):
        super(PointPolicyTransformer, self).__init__()
        
        # Define channel sizes and compute total output dimensions.
        channels = [3, 3, 4, 1, 3, 45]
        n_output_dims = sum(channels)  # 59
        
        # Load configuration from file (if needed for future extensions).
        with open("data.json") as f:
            config = json.load(f)
        
        # Project each input feature (scalar) to a d_model-dimensional embedding.
        # We treat the input as a sequence of 6 tokens, each originally 1-dimensional.
        self.input_proj = nn.Linear(1, d_model)
        
        # Create a learnable positional embedding for a fixed sequence length of 6.
        self.pos_embedding = nn.Parameter(torch.randn(n_input_dims, d_model))
        
        # Define a Transformer Encoder composed of several layers.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # After transformer processing, flatten the sequence (6 tokens * d_model) and map to n_output_dims.
        self.fc_out = nn.Linear(n_input_dims * d_model, n_output_dims)
        
        # Store channel sizes for later splitting.
        self.channels = channels

    def forward(self, x):
        # x shape: [N, 6]
        N = x.shape[0]
        # Reshape x to [N, 6, 1] so each scalar is treated as a token.
        x = x.unsqueeze(-1)  # shape: [N, 6, 1]
        
        # Project each token from 1 to d_model dimensions.
        x = self.input_proj(x)  # shape: [N, 6, d_model]
        
        # Add the positional embedding (broadcasted to batch dimension).
        x = x + self.pos_embedding.unsqueeze(0)  # shape: [N, 6, d_model]
        
        # Transformer expects input shape [sequence_length, batch_size, d_model].
        x = x.transpose(0, 1)  # shape: [6, N, d_model]
        
        # Process with the Transformer encoder.
        x = self.transformer_encoder(x)  # shape: [6, N, d_model]
        
        # Transpose back to [N, 6, d_model] and flatten.
        x = x.transpose(0, 1).reshape(N, -1)  # shape: [N, 6 * d_model]
        
        # Final linear layer to produce the output actions.
        actions = self.fc_out(x) / 1000000  # Scale down as in the original code.
        
        # Split the actions into the defined channel sizes.
        actions_split = torch.split(actions, self.channels, dim=-1)
        keys = ["_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest"]
        lr_adjust_dict = {k: {"action": a} for k, a in zip(keys, actions_split)}
        
        return lr_adjust_dict


class PointPolicyD(nn.Module):
    def __init__(self, n_input_dims=6):
        super(PointPolicyD, self).__init__()
        
        # Define channel sizes and compute total output dimensions.
        channels = [3, 3, 4, 1, 3, 45]
        n_output_dims = sum(channels)  # 3+3+4+1+3+45 = 59
        
        with open("data.json") as f:
            config = json.load(f)

        # Option 2: separate modules. Slower but more flexible.
        self.encoding = tcnn.Encoding(n_input_dims, config["encoding"])
        self.network = tcnn.Network(self.encoding.n_output_dims, n_output_dims, config["network"])
        self.policy_network = nn.Sequential(self.encoding, self.network)
        
        # Learnable log standard deviation for the Gaussian policy.
        #self.log_std = nn.Parameter(torch.zeros(n_output_dims))
        # Store channel sizes for splitting later.
        self.channels = channels
        
    def forward(self, x):
        # Forward pass through the policy network (shape: [N, n_input_dims] -> [N, n_output_dims])
        actions = self.policy_network(x)/1000000 # 1000000
        #print(torch.mean(torch.abs(actions)))

        # Split the outputs according to the defined channel sizes.
        actions_split = torch.split(actions, self.channels, dim=-1)
        
        keys = ["_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest"]
        lr_adjust_dict = {
            k: {"action": a} 
            for k, a in zip(keys, actions_split)
        }
        
        return lr_adjust_dict



class PointMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        """
        A simple MLP-based critic that takes a state representation (here, concatenated [xyz, scaling])
        and outputs a scalar value estimate.
        """
        super(PointMLP, self).__init__()
        #self.channels = [3, 3, 4, 1, 3, 45]
        self.channels =  [3, 3, 4, 1, 3, 45]
        n_output_dims = sum(self.channels)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, n_output_dims)
        
        
    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) + x 
        x = F.relu(self.fc3(x)) + x
        x = F.relu(self.fc4(x)) + x 
        x = F.relu(self.fc5(x))

        x = self.fc_out(x) / 10000000 #/ 10010000000
        actions_split = torch.split(x, self.channels, dim=-1)
        
        keys = ["_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest"]
        lr_adjust_dict = {
            k: {"action": a} 
            for k, a in zip(keys, actions_split)
        }
        
        return lr_adjust_dict


class PointPolicyV1(nn.Module):
    def __init__(self, n_input_dims=6):
        super(PointPolicyV1, self).__init__()
        
        # Define channel sizes and compute total output dimensions.
        channels = [3, 3, 4, 1, 3, 45]
        n_output_dims = sum(channels)  # 3+3+4+1+3+45 = 59
        
        with open("data_hash_small.json") as f:
            config = json.load(f)

        # Option 2: separate modules. Slower but more flexible.
        self.encoding = tcnn.Encoding(n_input_dims, config["encoding"])
        self.network = tcnn.Network(self.encoding.n_output_dims, n_output_dims, config["network"])
        self.policy_network = nn.Sequential(self.encoding, self.network)
        
        # Learnable log standard deviation for the Gaussian policy.
        self.log_std = nn.Parameter(torch.zeros(n_output_dims))
        # Store channel sizes for splitting later.
        self.channels = channels
        
    def forward(self, x):
        # Forward pass through the policy network (shape: [N, n_input_dims] -> [N, n_output_dims])
        means = self.policy_network(x)
        
        # Expand the learnable log_std to match the shape.
        std = torch.exp(self.log_std)         # shape: (n_output_dims,)
        std = std.unsqueeze(0).expand_as(means) # shape: (N, n_output_dims)
          
        # Sample from a Gaussian using the reparameterization trick.
        eps = torch.randn_like(means)
        actions = means + std * eps
        #actions = nn.functional.softplus(actions)
        #actions = 1 + 0.1* (torch.sigmoid(actions)-0.5) 
        
        #actions = 1 + 0.3* (torch.sigmoid(actions)-0.5)  # 0.7 - 1.3
        #actions = 1 + 0.5* (torch.sigmoid(actions)-0.5)  # 0.5 - 1.5
        
        # Compute log probabilities under the Gaussian.
        # log_probs = -0.5 * (((actions - means) / std) ** 2) - self.log_std - 0.5 * math.log(2 * math.pi)
        # log_probs = -0.5 * (((actions - means) / std) ** 2 + 2 * self.log_std + math.log(2 * math.pi))
        log_probs = -0.5 * (((actions - means) / std) ** 2) - self.log_std - 0.5 * math.log(2 * math.pi)




        # Split the outputs according to the defined channel sizes.
        actions_split = torch.split(actions, self.channels, dim=-1)
        log_probs_split = torch.split(log_probs, self.channels, dim=-1)
        
        keys = ["_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest"]
        lr_adjust_dict = {
            k: {"action": a, "log_prob": lp} 
            for k, a, lp in zip(keys, actions_split, log_probs_split)
        }
        
        return lr_adjust_dict
    
        
class PointPolicyV2(nn.Module):
    def __init__(self, n_input_dims=6):
        super(PointPolicyV2, self).__init__()
        
        # Define channel sizes and compute total output dimensions.
        channels = [3, 3, 4, 1, 3, 45]
        n_output_dims = sum(channels)  # 3+3+4+1+3+45 = 59
        
        with open("data_hash_small.json") as f:
            config = json.load(f)

        # Option 2: separate modules. Slower but more flexible.
        # self.encoding = tcnn.Encoding(n_input_dims, config["encoding"])
        # self.network = tcnn.Network(self.encoding.n_output_dims, n_output_dims, config["network"])
        # self.policy_network = nn.Sequential(self.encoding, self.network)
        
        
        #        self.encoding = tcnn.Encoding(n_input_dims, config["encoding"])
        self.network = tcnn.Network(n_input_dims, n_output_dims, config["network"])
        self.policy_network = self.network
        
        # Learnable log standard deviation for the Gaussian policy.
        self.log_std = nn.Parameter(torch.zeros(n_output_dims))
        # Store channel sizes for splitting later.
        self.channels = channels
        self.keys =  ["_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest"]
    def forward(self, x):
        # Forward pass through the policy network (shape: [N, n_input_dims] -> [N, n_output_dims])
        means = self.policy_network(x)
        
        # Expand the learnable log_std to match the shape.
        std = torch.exp(self.log_std)         # shape: (n_output_dims,)
        std = std.unsqueeze(0).expand_as(means) # shape: (N, n_output_dims)
          
        # Sample from a Gaussian using the reparameterization trick.
        eps = torch.randn_like(means)
        actions = means + std * eps 
        #actions = nn.functional.softplus(actions)
        #actions = 1 + 0.1* (torch.sigmoid(actions)-0.5) 
        
        #actions = 1 + 0.3* (torch.sigmoid(actions)-0.5)  # 0.7 - 1.3
        #actions = 1 + 0.5* (torch.sigmoid(actions)-0.5)  # 0.5 - 1.5
        
        log_probs = -0.5 * (((actions - means) / std) ** 2) - self.log_std - 0.5 * math.log(2 * math.pi)




        # Split the outputs according to the defined channel sizes.
        actions_split = torch.split(actions, self.channels, dim=-1)
        log_probs_split = torch.split(log_probs, self.channels, dim=-1)
        
        
        lr_adjust_dict = {
            k: {"action": a, "log_prob": lp} 
            for k, a, lp in zip(self.keys, actions_split, log_probs_split)
        }
        
        return lr_adjust_dict
    

class SimpleScalarPolicyV2(nn.Module):
    def __init__(self, keys, init_std: float = 0.1, clamp=False):
        super().__init__()
        self.keys = keys
        self.mu = nn.Parameter(torch.ones(len(keys)))           # μ = 1
        # σ = init_std ⇒ log σ = ln(init_std)
        self.log_std = nn.Parameter(
            torch.full((len(keys),), math.log(init_std))
        )
        self.clamp = clamp 
        
    def forward(self, batch_size: int = 1, device=None):
        """
        如果你真的不需要任何输入，把 forward 设计成接受 batch_size 就行；
        仍然兼容 torchscript / DDP 并且易于和采样器配合。
        """
        if device is None:
            device = self.mu.device


        mu  = self.mu.unsqueeze(0).expand(batch_size, -1).to(device)       # [B,6]
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu).to(device)  # [B,6]


        eps     = torch.randn_like(mu)          # [B,6]
        
        actions = mu + eps * std                # re‑parameterization trick
        
        # if self.clamp:
        #    actions = actions.clamp(0.9, 1.1)


        log_probs = -0.5 * ((eps) ** 2) - self.log_std - 0.5 * math.log(2 * math.pi)  # [B,6]


        out = {
            k: {
                "action": actions[:, i],        # shape [B]
                "log_prob": log_probs[:, i]     # shape [B]
            }
            for i, k in enumerate(self.keys)
        }
        return out


def squash_correction(log_prob, action):
    correction = torch.log(1 - action ** 2 + 1e-6)
    return log_prob - correction

class TanhScaledPolicy(nn.Module):
    def __init__(self, key_info_list):
        super().__init__()
        self.keys = [k for k, _, _, _ in key_info_list]
        self.mins = torch.tensor([min_v for _, _, min_v, _ in key_info_list])
        self.maxs = torch.tensor([max_v for _, _, _, max_v in key_info_list])
        init_mu = torch.tensor([init for _, init, _, _ in key_info_list])

        self.mu = nn.Parameter(init_mu)
        #self.log_std = nn.Parameter(torch.zeros(len(self.keys)) * 0.5)
        self.log_std = nn.Parameter(torch.ones(len(self.keys)) * (-1))

    def forward(self, prev_step_loss, batch_size=1, device=None):
        if device is None:
            device = self.mu.device

        mu = self.mu.unsqueeze(0).expand(batch_size, -1).to(device)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu).to(device)
        eps = torch.randn_like(mu)
        z = mu + eps * std

        a = torch.tanh(z)
        min_v = self.mins.to(device)
        max_v = self.maxs.to(device)
        scaled = 0.5 * (a + 1.0) * (max_v - min_v) + min_v

        log_prob = -0.5 * eps ** 2 - self.log_std - 0.5 * math.log(2 * math.pi)
        log_prob = squash_correction(log_prob, a)

        return {
            k: {"action": scaled[:, i], "log_prob": log_prob[:, i]}
            for i, k in enumerate(self.keys)
        }


# class ClampedGaussianPolicy(nn.Module):
#     def __init__(self, key_info_list):
#         super().__init__()
#         self.keys = [k for k, _, _, _ in key_info_list]
#         self.mins = torch.tensor([min_v for _, _, min_v, _ in key_info_list])
#         self.maxs = torch.tensor([max_v for _, _, _, max_v in key_info_list])
#         init_mu = torch.tensor([init for _, init, _, _ in key_info_list])

#         self.mu = nn.Parameter(init_mu)
#         self.log_std = nn.Parameter(torch.ones(len(self.keys)) * 0.5)

#     def forward(self, prev_step_loss, batch_size=1, device=None):
#         if device is None:
#             device = self.mu.device

#         mu = self.mu.unsqueeze(0).expand(batch_size, -1).to(device)
#         std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu).to(device)
#         eps = torch.randn_like(mu)
#         raw_action = mu + eps * std


#         min_v = self.mins.to(device).unsqueeze(0).expand_as(raw_action)
#         max_v = self.maxs.to(device).unsqueeze(0).expand_as(raw_action)
#         action = torch.clamp(raw_action, min=min_v, max=max_v)


#         log_prob = -0.5 * eps ** 2 - self.log_std - 0.5 * math.log(2 * math.pi)

#         return {
#             k: {"action": action[:, i], "log_prob": log_prob[:, i]}
#             for i, k in enumerate(self.keys)
#         }


class TanhScaledGaussianPolicyV2(nn.Module):
    def __init__(self, key_info_list, init_std=-3.0, max_std=1.0, std_schedule_steps=30000):
        super().__init__()
        self.keys = [k for k, _, _, _ in key_info_list]
        self.mins = torch.tensor([min_v for _, _, min_v, _ in key_info_list])
        self.maxs = torch.tensor([max_v for _, _, _, max_v in key_info_list])
        init_mu = torch.tensor([init for _, init, _, _ in key_info_list])

        self.base_mu = nn.Parameter(init_mu)


        self.init_log_std = init_std
        self.max_log_std = math.log(max_std)
        self.std_schedule_steps = std_schedule_steps


        self.loss_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.keys))
        )
        nn.init.zeros_(self.loss_encoder[-1].weight)
        nn.init.zeros_(self.loss_encoder[-1].bias)

    def forward(self, prev_step_loss, step, batch_size=1, device=None):
        if device is None:
            device = self.base_mu.device


        if not torch.is_tensor(prev_step_loss):
            prev_step_loss = torch.tensor(prev_step_loss, dtype=torch.float32, device=device)
        if not torch.is_tensor(step): 
            step = torch.tensor(step, dtype=torch.float32, device=device)


        loss_input = torch.stack((prev_step_loss, 1 - step))
        loss_input = loss_input.view(1, 2).expand(batch_size, -1)
        loss_offset = self.loss_encoder(loss_input)

        mu = self.base_mu.unsqueeze(0).expand(batch_size, -1) + loss_offset


        progress = torch.clamp(step / self.std_schedule_steps, 0.0, 1.0)
        log_std_now = self.init_log_std + (self.max_log_std - self.init_log_std) * progress
        log_std_now = log_std_now.detach()
        std = torch.exp(log_std_now).unsqueeze(0).expand_as(mu)

        eps = torch.randn_like(mu)
        raw_action = mu + eps * std
        bounded_action = torch.tanh(raw_action)


        min_v = self.mins.to(device).unsqueeze(0).expand_as(bounded_action)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(bounded_action)
        action = 0.5 * (bounded_action + 1) * (max_v - min_v) + min_v


        log_prob = -0.5 * eps**2 - log_std_now - 0.5 * math.log(2 * math.pi)
        correction = torch.log(1 - bounded_action**2 + 1e-6)
        log_prob = log_prob - correction

        return {
            k: {
                "action": action[:, i],
                "log_prob": log_prob[:, i]
            }
            for i, k in enumerate(self.keys)
        }
        


class TanhScaledGaussianPolicy(nn.Module):
    def __init__(self, key_info_list,  init_std=-1.0):
        super().__init__()
        self.keys = [k for k, _, _, _ in key_info_list]
        self.mins = torch.tensor([min_v for _, _, min_v, _ in key_info_list])
        self.maxs = torch.tensor([max_v for _, _, _, max_v in key_info_list])
        init_mu = torch.tensor([init for _, init, _, _ in key_info_list])

        self.base_mu = nn.Parameter(init_mu)
        self.log_std = nn.Parameter(torch.ones(len(self.keys)) * (init_std))

        self.loss_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.keys))
        )
        nn.init.zeros_(self.loss_encoder[-1].weight)
        nn.init.zeros_(self.loss_encoder[-1].bias)


    def forward(self, prev_step_loss, step, batch_size=1, device=None):
        if device is None:
            device = self.base_mu.device

        # Convert loss to tensor
        if not torch.is_tensor(prev_step_loss):
            prev_step_loss = torch.tensor(prev_step_loss, dtype=torch.float32, device=device)
            
        if not torch.is_tensor(step): 
            step =  torch.tensor(step, dtype=torch.float32, device=device)
        
        
        loss_input = torch.stack((prev_step_loss, 1-step))
        loss_input = loss_input.view(1, 2).expand(batch_size, -1)  # [B,1]
        
        loss_offset = self.loss_encoder(loss_input)                   # [B,N]

        mu = self.base_mu.unsqueeze(0).expand(batch_size, -1) + loss_offset  # [B,N]
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)
        
        

        eps = torch.randn_like(mu)
        

        raw_action = mu + eps * std
        
        
        
        
        
        bounded_action = torch.tanh(raw_action)  # [-1, 1]

        # Scale to [min, max]
        min_v = self.mins.to(device).unsqueeze(0).expand_as(bounded_action)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(bounded_action)
        action = 0.5 * (bounded_action + 1) * (max_v - min_v) + min_v


        log_prob = -0.5 * eps**2 - self.log_std - 0.5 * math.log(2 * math.pi)
        correction = torch.log(1 - bounded_action**2 + 1e-6)
        log_prob = log_prob - correction

        return {
            k: {
                "action": action[:, i],
                "log_prob": log_prob[:, i]
            }
            for i, k in enumerate(self.keys)
        }


class TanhScaledGaussianPolicyStateLess(nn.Module):
    def __init__(self, key_info_list, init_std=1.0):
        super().__init__()
        self.keys = [k for k, _, _, _ in key_info_list]
        self.mins = torch.tensor([min_v for _, _, min_v, _ in key_info_list])
        self.maxs = torch.tensor([max_v for _, _, _, max_v in key_info_list])
        init_mu = torch.tensor([init for _, init, _, _ in key_info_list])

        self.base_mu = nn.Parameter(init_mu)

        self.log_std = nn.Parameter(torch.ones(len(self.keys))*init_std)



    def forward(self, step=None, batch_size=1, device=None):
        if device is None:
            device = self.base_mu.device
        
        mu = self.base_mu.unsqueeze(0).expand(batch_size, -1) # [B,N]
        
        with torch.no_grad():
            self.log_std.clamp_(-4.0, 0.0)
    
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)

        eps = torch.randn_like(mu)
        raw_action = mu + eps * std
        bounded_action = torch.tanh(raw_action)  # [-1, 1]

        # Scale to [min, max]
        min_v = self.mins.to(device).unsqueeze(0).expand_as(bounded_action)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(bounded_action)
        action = 0.5 * (bounded_action + 1) * (max_v - min_v) + min_v


        log_prob = -0.5 * eps**2 - self.log_std - 0.5 * math.log(2 * math.pi)
        correction = torch.log(1 - bounded_action**2 + 1e-6)
        log_prob = log_prob - correction

        return {
            k: {
                "action": action[:, i],
                "log_prob": log_prob[:, i]
            }
            for i, k in enumerate(self.keys)
        }
        
class ClampedGaussianPolicy(nn.Module):
    def __init__(self, key_info_list, init_std):
        super().__init__()
        self.keys = [k for k, _, _, _ in key_info_list]
        #self.scaler = 5000 #  0.00016, 1e-5, 5e-4
        self.mins = torch.tensor([min_v for _, _, min_v, _ in key_info_list])
        self.maxs = torch.tensor([max_v for _, _, _, max_v in key_info_list])
        init_mu = torch.tensor([init for _, init, _, _ in key_info_list])

        self.base_mu = nn.Parameter(init_mu )
        self.log_std = nn.Parameter(torch.ones(len(self.keys)) * (init_std))


        self.loss_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.keys))
        )
        
        nn.init.zeros_(self.loss_encoder[-1].weight)
        nn.init.zeros_(self.loss_encoder[-1].bias)
        

    def forward(self, prev_step_loss, step, batch_size=1, device=None):
        if device is None:
            device = self.base_mu.device

        if not torch.is_tensor(prev_step_loss):
            prev_step_loss = torch.tensor(prev_step_loss, dtype=torch.float32, device=device)
        if not torch.is_tensor(step):
            step = torch.tensor(step, dtype=torch.float32, device=device)

        loss_input = torch.stack((prev_step_loss, 1 - step))
        loss_input = loss_input.view(1, 2).expand(batch_size, -1)  # [B, 2]

        loss_offset = self.loss_encoder(loss_input)  # [B, N]

        mu = (self.base_mu).unsqueeze(0).expand(batch_size, -1) #+ loss_offset  # [B, N]
        
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)

        eps = torch.randn_like(mu) #/ self.scaler
        raw_action = ((mu + eps * std)).detach()  # 
        

        # Clamp
        min_v = self.mins.to(device).unsqueeze(0).expand_as(raw_action)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(raw_action)
        action = torch.clamp(raw_action, min=min_v, max=max_v)
        # Log prob (Gaussian, no transform)
        log_prob = -0.5 * ((raw_action - mu) / std) ** 2 - self.log_std - 0.5 * math.log(2 * math.pi)

        return {
            k: {"action": action[:, i], "log_prob": log_prob[:, i]}
            for i, k in enumerate(self.keys)
        }


class SimpleScalarPolicy(nn.Module):
    """
    Policy 完全由可学习的 6 个 μ 和 6 个 log σ 组成，和环境 state 无关。
    每次调用 forward() 返回一批动作样本及其对数概率。
    """
    def __init__(self, keys):
        super().__init__()

        self.keys = keys
        self.mu      = nn.Parameter(torch.ones(len(self.keys)))   # shape [6]
        self.log_std = nn.Parameter(torch.zeros(len(self.keys)))  # shape [6]

    def forward(self, batch_size: int = 1, device=None):
        """
        如果你真的不需要任何输入，把 forward 设计成接受 batch_size 就行；
        仍然兼容 torchscript / DDP 并且易于和采样器配合。
        """
        if device is None:
            device = self.mu.device


        mu  = self.mu.unsqueeze(0).expand(batch_size, -1).to(device)       # [B,6]
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu).to(device)  # [B,6]


        eps     = torch.randn_like(mu)          # [B,6]
        actions = (mu + eps * std)#.detach()                # re‑parameterization trick


        log_probs = -0.5 * (((actions -mu)/std) ** 2) - self.log_std - 0.5 * math.log(2 * math.pi)  # [B,6]


        out = {
            k: {
                "action": actions[:, i],        # shape [B]
                "log_prob": log_probs[:, i]     # shape [B]
            }
            for i, k in enumerate(self.keys)
        }
        return out




class SimpleScalarPolicyClampSTD(nn.Module):
    """
    Policy 完全由可学习的 6 个 μ 和 6 个 log σ 组成，和环境 state 无关。
    每次调用 forward() 返回一批动作样本及其对数概率。
    """
    def __init__(self, keys, init_std=-1.0, clamp=False, detach=True):
        super().__init__()
        
        self.keys = keys
        self.mu      = nn.Parameter(torch.ones(len(self.keys)))   # shape [6]
        self.log_std = nn.Parameter(torch.ones(len(self.keys)) * init_std)  # shape [6]
        self.clamp = clamp
        self.detach = detach

    def forward(self, batch_size: int = 1, device=None):
        """
        如果你真的不需要任何输入，把 forward 设计成接受 batch_size 就行；
        仍然兼容 torchscript / DDP 并且易于和采样器配合。
        """
        if device is None:
            device = self.mu.device


        mu  = self.mu.unsqueeze(0).expand(batch_size, -1).to(device)       # [B,6]
        if self.clamp:
            log_std = torch.clamp(self.log_std, -20, 1)
            std = torch.exp(log_std).unsqueeze(0).expand_as(mu).to(device)
        else:
            std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu).to(device)  # [B,6]
        
          

        eps     = torch.randn_like(mu)          # [B,6]
        if self.detach :
            actions = (mu + eps * std)              # re‑parameterization trick
            actions = actions.detach()
        else:
            actions = (mu + eps * std)


        log_probs = -0.5 * (((actions -mu)/std) ** 2) - self.log_std - 0.5 * math.log(2 * math.pi)  # [B,6]


        
        out = {
                k: {
                    "action": actions[:, i],        # shape [B]
                    "log_prob": log_probs[:, i]     # shape [B]
                }
                for i, k in enumerate(self.keys)
            }
        return out
        
    
    def freeze_parameters(self):
        self.mu.requires_grad = False
        self.log_std.requires_grad = False

    def unfreeze_parameters(self):
        self.mu.requires_grad = True
        self.log_std.requires_grad = True



class LogNormalLRPolicy(nn.Module):
    """
    直接学习 log-factor (z)，并返回 factor = exp(z)。
    - mu_z 和 log_std_z 仍是 6 个可学习参数
    - 初始 μ≈0 ⇒ factor≈1
    - 初始 σ≈0.1‒0.2 ⇒ factor 落在 ≈[0.8, 1.25]
    """

    def __init__(self, keys, init_std=-2.0, clamp=False, detach=True):
        """
        init_std = -2.0  →  σ = exp(-2) ≈ 0.14
        """
        super().__init__()
        self.keys   = keys
        self.mu      = nn.Parameter(torch.zeros(len(keys)))               # μ_z
        self.log_std = nn.Parameter(torch.ones(len(keys)) * init_std)     # log σ_z
        self.clamp  = clamp
        self.detach = detach

    def forward(self, batch_size: int = 1, device=None):
        if device is None:
            device = self.mu.device


        mu  = self.mu.unsqueeze(0).expand(batch_size, -1).to(device)      # [B,6]
        if self.clamp:
            log_std = torch.clamp(self.log_std, -10, -1)                # σ ∈ (0.00005, 0.37)
            std = torch.exp(log_std).unsqueeze(0).expand_as(mu).to(device)
        else:
            std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu).to(device)

        eps  = torch.randn_like(mu)
        z    = mu + eps * std                                               # log-factor
        if self.detach:
            z = z.detach()

        factor = torch.exp(z)



        log_prob = -0.5 * ((z - mu) / std) ** 2\
                   - self.log_std - 0.5 * math.log(2 * math.pi) - z       # [B,6]


        out = {k: {"action": factor[:, i], "log_prob": log_prob[:, i]}
               for i, k in enumerate(self.keys)}
        return out


class SimpleScalarPolicyClampSTDClamp(nn.Module):
    """
    Policy 完全由可学习的 6 个 μ 和 6 个 log σ 组成，和环境 state 无关。
    每次调用 forward() 返回一批动作样本及其对数概率。
    """
    def __init__(self, key_info_list, init_std=0, clamp=False, detach=True):
        super().__init__()
        
        
        self.mins = torch.tensor([min_v / default for name, default, min_v, max_v in key_info_list])
        self.maxs = torch.tensor([max_v / default for name, default, min_v, max_v in key_info_list])
        
        

        self.keys = [name for name, _, _, _ in key_info_list]
        self.log_std = nn.Parameter(torch.ones(len(self.keys)) * init_std)  # shape [6]
        self.mu = nn.Parameter(torch.ones(len(self.keys)))

        
        self.clamp = clamp
        self.detach = detach

    def forward(self, batch_size: int = 1, device=None):
        """
        如果你真的不需要任何输入，把 forward 设计成接受 batch_size 就行；
        仍然兼容 torchscript / DDP 并且易于和采样器配合。
        """
        if device is None:
            device = self.mu.device


        mu  = self.mu.unsqueeze(0).expand(batch_size, -1).to(device)       # [B,6]
        if self.clamp:
            log_std = torch.clamp(self.log_std, -20, 1)
            std = torch.exp(log_std).unsqueeze(0).expand_as(mu).to(device)
        else:
            std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu).to(device)  # [B,6]
        
          

        eps     = torch.randn_like(mu)          # [B,6]
        if self.detach :
            raw_action = (mu + eps * std).detach()                # re‑parameterization trick
        else:
            raw_action = (mu + eps * std)
            
        


        log_probs = -0.5 * (((raw_action -mu)/std) ** 2) - self.log_std - 0.5 * math.log(2 * math.pi)  # [B,6]
        


        min_v = self.mins.to(device).unsqueeze(0).expand_as(raw_action)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(raw_action)
        action = torch.clamp(raw_action, min=min_v, max=max_v)



        
        if batch_size == 1:
            out = {
                k: {
                    "action": action[:, i],        # shape [B]
                    "log_prob": log_probs[:, i]     # shape [B]
                }
                for i, k in enumerate(self.keys)
            }
            return out
        else:
            rt = []
            for bidx in range(batch_size):
                out = {
                k: {
                    "action": action[bidx:bidx+1, i],        # shape [B]
                    "log_prob": log_probs[bidx:bidx+1, i]     # shape [B]
                }
                for i, k in enumerate(self.keys)
                }
                rt.append(out)
            return rt
    
    def freeze_parameters(self):
        self.mu.requires_grad = False
        self.log_std.requires_grad = False

    def unfreeze_parameters(self):
        self.mu.requires_grad = True
        self.log_std.requires_grad = True

# log_alpha = torch.zeros(1, requires_grad=True, device=device)
# alpha = log_alpha.exp()

# # optim
# alpha_optimizer = torch.optim.Adam([log_alpha], lr=alpha_lr)


# alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()


# alpha_optimizer.zero_grad()
# alpha_loss.backward()
# alpha_optimizer.step()




class Sanwich(nn.Module):
    """
    Policy 完全由可学习的 6 个 μ 和 6 个 log σ 组成，和环境 state 无关。
    每次调用 forward() 返回一批动作样本及其对数概率。
    """
    def __init__(self, key_info_list, init_std=0, clamp=False, detach=True):
        super().__init__()
        
        
        self.mins = torch.tensor([min_v / default for name, default, min_v, max_v in key_info_list])
        self.maxs = torch.tensor([max_v / default for name, default, min_v, max_v in key_info_list])
        
        

        self.keys = [name for name, _, _, _ in key_info_list]
        self.log_std = nn.Parameter(torch.ones(len(self.keys)) * init_std)  # shape [6]
        self.mu = nn.Parameter(torch.ones(len(self.keys)))

        self.loss_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.keys))
        )
        
        nn.init.zeros_(self.loss_encoder[-1].weight)
        nn.init.zeros_(self.loss_encoder[-1].bias)
        self.clamp = clamp
        self.detach = detach

    
    def forward(self, prev_step_loss, step, batch_size=1, device=None):
        if device is None:
            device = self.mu.device

        if not torch.is_tensor(prev_step_loss):
            prev_step_loss = torch.tensor(prev_step_loss, dtype=torch.float32, device=device)
        if not torch.is_tensor(step):
            step = torch.tensor(step, dtype=torch.float32, device=device)

        loss_input = torch.stack((prev_step_loss, 1 - step))
        loss_input = loss_input.view(1, 2).expand(batch_size, -1)  # [B, 2]

        loss_offset = self.loss_encoder(loss_input)  # [B, N]


        if device is None:
            device = self.mu.device


        mu  = self.mu.unsqueeze(0).expand(batch_size, -1).to(device)   + loss_offset    # [B,6]
        if self.clamp:
            log_std = torch.clamp(self.log_std, -20, 1)
            std = torch.exp(log_std).unsqueeze(0).expand_as(mu).to(device)
        else:
            std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu).to(device)  # [B,6]
        
          

        eps     = torch.randn_like(mu)          # [B,6]
        if self.detach :
            raw_action = (mu + eps * std).detach()                # re‑parameterization trick
        else:
            raw_action = (mu + eps * std)
            
        


        log_probs = -0.5 * (((raw_action -mu)/std) ** 2) - self.log_std - 0.5 * math.log(2 * math.pi)  # [B,6]
        


        min_v = self.mins.to(device).unsqueeze(0).expand_as(raw_action)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(raw_action)
        action = torch.clamp(raw_action, min=min_v, max=max_v)



        
        if batch_size == 1:
            out = {
                k: {
                    "action": action[:, i],        # shape [B]
                    "log_prob": log_probs[:, i]     # shape [B]
                }
                for i, k in enumerate(self.keys)
            }
            return out
        else:
            rt = []
            for bidx in range(batch_size):
                out = {
                k: {
                    "action": action[bidx:bidx+1, i],        # shape [B]
                    "log_prob": log_probs[bidx:bidx+1, i]     # shape [B]
                }
                for i, k in enumerate(self.keys)
                }
                rt.append(out)
            return rt

class ScalarPolicy(nn.Module):
    def __init__(self, n_input_dims=6):
        super(ScalarPolicy, self).__init__()
        # Define parameter keys.
        self.keys = ["xyz", "scaling", "rotation", "opacity", "f_dc", "f_rest"]
        n_output_dims = len(self.keys)  # one scalar per key (6 total)

        # Load configuration for the encoding/network.
        with open("data_hash_small.json") as f:
            config = json.load(f)

        # Build the network: first, an encoding module, then the main network producing 6 outputs.
        self.encoding = tcnn.Encoding(n_input_dims, config["encoding"])
        self.network = tcnn.Network(self.encoding.n_output_dims, n_output_dims, config["network"])
        self.policy_network = nn.Sequential(self.encoding, self.network)
        
        # Learnable log standard deviation for the Gaussian policy (one per key).
        self.log_std = nn.Parameter(torch.zeros(n_output_dims))
        
    def forward(self, x):
        """
        Expects x of shape [n_input_dims] or [1, n_input_dims]. In the first case, the input is unsqueezed
        to add a batch dimension. After processing, the batch dimension is removed so that each key's entry has shape [1].
        """
      

        # Forward pass: output shape: [N, 6] where N is batch size (should be 1)
        means = self.policy_network(x)
        means_aggregated = torch.mean(means, dim=0, keepdim=True) + 1.0 # shape: [1, 6]
        
        # Compute standard deviation from the learnable log_std (shape [6]) and expand to shape [1,6].
        std = torch.exp(self.log_std)         # shape: [6]
        std = std.unsqueeze(0)                # shape: [1, 6]
        
        # Sample one scalar per key using the reparameterization trick.
        eps = torch.randn_like(means_aggregated)  # shape: [1, 6]
        actions = means_aggregated + std * eps      # shape: [1, 6]
        
        # Compute log probabilities for the sampled actions.
        log_probs = -0.5 * (((actions - means_aggregated) / std) ** 2) - self.log_std -\
                    0.5 * math.log(2 * math.pi)  # shape: [1, 6]
        
        # For each key, extract the scalar (a tensor of shape [1]) for action and log probability.
        lr_adjust_dict = {
            k: {"action": actions[0, i:i+1], "log_prob": log_probs[0, i:i+1]}
            for i, k in enumerate(self.keys)
        }
        
        return lr_adjust_dict


# class LatentPolicy(nn.Module):
#     def __init__(self, n_input_dims, n_output_dims):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(n_input_dims, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128)
#         )

#         self.policy_net = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, n_output_dims)
#         )

#         self.log_std = nn.Parameter(torch.zeros(n_output_dims))

#     def forward(self, condition):
#         noise = torch.randn_like(condition)  # same dim
#         latent = condition + noise           # or concatenate: torch.cat([condition, noise], dim=-1)

#         h = self.encoder(latent)
#         mean = self.policy_net(h) + 1.0      # offset  

#         std = torch.exp(self.log_std).unsqueeze(0)  # shape [1, D]
#         eps = torch.randn_like(mean)

#         actions = mean + std * eps
#         log_probs = -0.5 * (((actions - mean) / std) ** 2) - self.log_std - 0.5 * math.log(2 * math.pi)

#         return actions, log_probs




class LatentPolicy(nn.Module):
    def __init__(self, n_input_dims=6, latent_dim=6):
        super(LatentPolicy, self).__init__()
        self.keys = ["xyz", "scaling", "rotation", "opacity", "f_dc", "f_rest"]
        n_output_dims = len(self.keys)

        with open("data_hash_small.json") as f:
            config = json.load(f)


        self.encoding = tcnn.Encoding(n_input_dims, config["encoding"])
        self.network = tcnn.Network(self.encoding.n_output_dims, 128, config["network"])


        self.latent_fc = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, n_output_dims)
        )

        self.log_std = nn.Parameter(torch.zeros(n_output_dims))

    def forward(self, condition):
        """
        condition: tensor of shape [1, n_input_dims] 或 [batch_size, n_input_dims]
        """
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)


        encoded = self.encoding(condition)        # [1, D']
        encoded = self.network(encoded)           # [1, 128]


        noise = torch.randn_like(encoded)
        latent = torch.cat([encoded, noise], dim=-1)  # [1, 128 + latent_dim]
        latent = latent.float()

        means = self.latent_fc(latent)            # [1, 6]
        means = means.mean(0, keepdim=True) 
        means_aggregated = means + 1.0


        std = torch.exp(self.log_std).unsqueeze(0)  # [1, 6]
        eps = torch.randn_like(means_aggregated)    # [1, 6]

        # reparameterization trick
        actions = means_aggregated + std * eps      # [1, 6]


        log_probs = -0.5 * (((actions - means_aggregated) / std) ** 2) - self.log_std -\
                    0.5 * math.log(2 * math.pi)      # [1, 6]


        lr_adjust_dict = {
            k: {"action": actions[0, i:i+1], "log_prob": log_probs[0, i:i+1]}
            for i, k in enumerate(self.keys)
        }

        return lr_adjust_dict





class ScalarPolicyNormed(nn.Module):
    def __init__(self, n_input_dims=6):
        super(ScalarPolicyNormed, self).__init__()
        # Define parameter keys.
        self.keys = ["xyz", "scaling", "rotation", "opacity", "f_dc", "f_rest"]
        n_output_dims = len(self.keys)  # one scalar per key (6 total)

        # Load configuration for the encoding/network.
        with open("data_hash_small.json") as f:
            config = json.load(f)
            
        self.layer_norm = nn.LayerNorm(n_input_dims)

        # Build the network: first, an encoding module, then the main network producing 6 outputs.
        self.encoding = tcnn.Encoding(n_input_dims, config["encoding"])
        self.network = tcnn.Network(self.encoding.n_output_dims, n_output_dims, config["network"])
        self.policy_network = nn.Sequential(self.encoding, self.network)
        
        # Learnable log standard deviation for the Gaussian policy (one per key).
        self.log_std = nn.Parameter(torch.zeros(n_output_dims))
        
    def forward(self, x):
        """
        Expects x of shape [n_input_dims] or [1, n_input_dims]. In the first case, the input is unsqueezed
        to add a batch dimension. After processing, the batch dimension is removed so that each key's entry has shape [1].
        """
      

        # Forward pass: output shape: [N, 6] where N is batch size (should be 1)
        x = self.layer_norm(x)
        means = self.policy_network(x)
        means_aggregated = torch.mean(means, dim=0, keepdim=True) + 1.0 # shape: [1, 6]
        
        # Compute standard deviation from the learnable log_std (shape [6]) and expand to shape [1,6].
        std = torch.exp(self.log_std)         # shape: [6]
        std = std.unsqueeze(0)                # shape: [1, 6]
        
        # Sample one scalar per key using the reparameterization trick.
        eps = torch.randn_like(means_aggregated)  # shape: [1, 6]
        actions = means_aggregated + std * eps      # shape: [1, 6]
        
        # Compute log probabilities for the sampled actions.
        log_probs = -0.5 * (((actions - means_aggregated) / std) ** 2) - self.log_std -\
                    0.5 * math.log(2 * math.pi)  # shape: [1, 6]
        
        # For each key, extract the scalar (a tensor of shape [1]) for action and log probability.
        lr_adjust_dict = {
            k: {"action": actions[0, i:i+1], "log_prob": log_probs[0, i:i+1]}
            for i, k in enumerate(self.keys)
        }
        
        return lr_adjust_dict
    
class PointPolicyV3(nn.Module):
    def __init__(self, n_input_dims=6):
        super(PointPolicyV3, self).__init__()
        
        # Define channel sizes and compute total output dimensions.
        channels = [3, 3, 4, 1, 3, 45]
        n_output_dims = sum(channels)  # 3+3+4+1+3+45 = 59
        
        with open("data_hash_small.json") as f:
            config = json.load(f)

        # Option 2: separate modules. Slower but more flexible.
        #self.encoding = tcnn.Encoding(n_input_dims, config["encoding"])
        self.network = tcnn.Network(n_input_dims, n_output_dims, config["network"])
        self.policy_network = self.network
        
        # Learnable log standard deviation for the Gaussian policy.
        self.log_std = nn.Parameter(torch.zeros(n_output_dims))
        # Store channel sizes for splitting later.
        self.channels = channels
        
    def forward(self, x):
        # Forward pass through the policy network (shape: [N, n_input_dims] -> [N, n_output_dims])
        means = self.policy_network(x)
        
        # Expand the learnable log_std to match the shape.
        std = torch.exp(self.log_std)         # shape: (n_output_dims,)
        std = std.unsqueeze(0).expand_as(means) # shape: (N, n_output_dims)
          
        # Sample from a Gaussian using the reparameterization trick.
        eps = torch.randn_like(means)
        actions = means + std * eps
        #actions = nn.functional.softplus(actions)
        #actions = 1 + 0.1* (torch.sigmoid(actions)-0.5) 
        
        #actions = 1 + 0.3* (torch.sigmoid(actions)-0.5)  # 0.7 - 1.3
        #actions = 1 + 0.5* (torch.sigmoid(actions)-0.5)  # 0.5 - 1.5
        
        # Compute log probabilities under the Gaussian.
        # log_probs = -0.5 * (((actions - means) / std) ** 2) - self.log_std - 0.5 * math.log(2 * math.pi)
        # log_probs = -0.5 * (((actions - means) / std) ** 2 + 2 * self.log_std + math.log(2 * math.pi))
        log_probs = -0.5 * (((actions - means) / std) ** 2) - self.log_std - 0.5 * math.log(2 * math.pi)




        # Split the outputs according to the defined channel sizes.
        actions_split = torch.split(actions, self.channels, dim=-1)
        log_probs_split = torch.split(log_probs, self.channels, dim=-1)
        
        keys = ["_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest"]
        lr_adjust_dict = {
            k: {"action": a, "log_prob": lp} 
            for k, a, lp in zip(keys, actions_split, log_probs_split)
        }
        
        return lr_adjust_dict

# class PointPolicy(nn.Module):
#     def __init__(self, n_input_dims=6, config_path="data.json"):
#         super(PointPolicy, self).__init__()
        
#         self.channels = [3, 3, 4, 1, 3, 45]
#         n_output_dims = sum(self.channels)
        
#         try:
#             with open(config_path) as f:
#                 config = json.load(f)
#         except FileNotFoundError:
#             raise FileNotFoundError(f"Config file not found at {config_path}")

#         self.encoding = tcnn.Encoding(n_input_dims, config["encoding"])
#         self.network = tcnn.Network(self.encoding.n_output_dims, n_output_dims, config["network"])
#         self.policy_network = nn.Sequential(self.encoding, self.network)
        
#         self.log-std = nn.Parameter(torch.zeros(n_output_dims))
        
#     def forward(self, x):
#         # Get raw means from the network
#         raw_means = self.policy_network(x)
        
#         # Apply softplus to ensure positive means
#         means = nn.functional.softplus(raw_means)
        
#         # Compute standard deviations
#         std = torch.exp(self.log_std)
#         std = std.unsqueeze(0).expand_as(means)
        
#         # Sample actions (will be positive due to positive means and std)
#         eps = torch.randn_like(means)
#         actions = means + std * eps
#         # Ensure actions are positive by applying softplus again
#         actions = nn.functional.softplus(actions)
        
#         # Compute log probabilities (adjusted for softplus transformation)
#         # Note: This is an approximation as the exact log_prob is more complex
#         log_probs = -0.5 * (((actions - means) / std) ** 2 + 2 * self.log_std + math.log(2 * math.pi))
        
#         # Split outputs
#         actions_split = torch.split(actions, self.channels, dim=-1)
#         log_probs_split = torch.split(log_probs, self.channels, dim=-1)
        
#         keys = ["_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest"]
#         lr_adjust_dict = {
#             k: {"action": a, "log_prob": lp}
#             for k, a, lp in zip(keys, actions_split, log_probs_split)
#         }
        
#         return lr_adjust_dict



class AttentionPooling(nn.Module):
    """
    输入 : 点特征 tensor  F ∈ ℝ^{N×C}
    输出 : 聚合后的向量     g ∈ ℝ^{1×C}

    公式 :
        q  = W_q ·  mean(F)            # learnable query (或直接学一个参数)
        aᵢ = softmax( (q · fᵢ) / √C )  # 点权重
        g  = Σ aᵢ · fᵢ
    """
    def __init__(self, dim, use_learnable_query=False):
        super().__init__()
        self.scale = dim ** -0.5                    # 1/√C

        if use_learnable_query:

            self.q = nn.Parameter(torch.zeros(1, dim))
        else:

            self.q_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, F):                           # F: [N, C]
        if hasattr(self, "q"):                      # learnable global query
            q = self.q                             # [1, C]
        else:
            q = self.q_proj(F.mean(0, keepdim=True))  # [1, C]


        attn = torch.softmax(torch.matmul(q, F.T) * self.scale, dim=-1)

        g = torch.matmul(attn, F)                  # [1, C]
        return g
    
class ScalarPolicyAtten(nn.Module):
    def __init__(self, n_input_dims=3, hidden_dim=64, heads=False):
        super().__init__()
        self.keys = ["xyz", "scaling", "rotation",
                     "opacity", "f_dc", "f_rest"]
        n_out = len(self.keys)


        with open("data_hash_small.json") as f:
            cfg = json.load(f)
        self.encoding = tcnn.Encoding(n_input_dims, cfg["encoding"])
        self.point_mlp = tcnn.Network(self.encoding.n_output_dims,
                                      hidden_dim, cfg["network"])


        self.pool = AttentionPooling(hidden_dim, use_learnable_query=heads)

        # ---- policy head --------------------------------------------------
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_out)
        )
        self.log_std = nn.Parameter(torch.zeros(n_out))

    def forward(self, pts):           # pts: [N, 3]
        f = self.point_mlp(self.encoding(pts)).float()     # [N, H]
        g = self.pool(f)                           # [1, H]  ← Attention pooling
        means = self.policy_head(g) + 1.0          # [1, 6]

        std = torch.exp(self.log_std)              # [6]
        eps = torch.randn_like(means)              # [1, 6]
        actions = means + std * eps
        logp = -0.5 * (eps**2) - self.log_std - 0.5*math.log(2*math.pi)

        return {k: {"action": actions[0,i], "log_prob": logp[0,i]}
                for i, k in enumerate(self.keys)}



class PGPEPolicy(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.mu      = nn.Parameter(torch.zeros(action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), -2.0))
    def sample(self):
        eps = torch.randn_like(self.mu)
        action = self.mu + torch.exp(self.log_std) * eps
        log_prob = (-0.5 * eps.pow(2) - self.log_std
                    - 0.5 * math.log(2*math.pi)).sum()
        return action, log_prob
    
    



class LogScaledGaussianPolicy(nn.Module):
    def __init__(self, key_info_list, init_std=-1.0, init_weight=-0.5,beta=0.01):
        super().__init__()
        self.keys = [k for k, _, _, _ in key_info_list]
        self.mins = torch.tensor([min_v for _, _, min_v, _ in key_info_list])
        self.maxs = torch.tensor([max_v for _, _, _, max_v in key_info_list])
        init_mu = torch.tensor([init for _, init, _, _ in key_info_list])

        self.base_mu = nn.Parameter(torch.log10(init_mu))
        self.log_std = nn.Parameter(torch.ones(len(self.keys)) * init_std)

        self.loss_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.keys))
        )
        nn.init.zeros_(self.loss_encoder[-1].weight)
        nn.init.zeros_(self.loss_encoder[-1].bias)
        self.beta = beta 
        # with torch.no_grad():
        #     self.loss_encoder[-1].bias.copy_(torch.ones(len(self.keys)) * init_weight)

    def forward(self, prev_step_loss, step, batch_size=1, device=None):
        if device is None:
            device = self.base_mu.device

        if not torch.is_tensor(prev_step_loss):
            prev_step_loss = torch.tensor(prev_step_loss, dtype=torch.float32, device=device)
        if not torch.is_tensor(step):
            step = torch.tensor(step, dtype=torch.float32, device=device)

        # loss_input = torch.stack((prev_step_loss, 1 - step))
        # loss_input = loss_input.view(1, 2).expand(batch_size, -1)  # [B, 2]

        # # Dynamic offset from prev loss
        # loss_offset = self.loss_encoder(loss_input)  # [B, N]

        # Compute raw Gaussian sample in log-space
        mu = self.base_mu.unsqueeze(0).expand(batch_size, -1) #+ loss_offset  # [B, N]
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)  # [B, N]

        eps = torch.randn_like(mu) * self.beta 
        raw_action = (mu + eps * std).detach()  # Sample in log space (log_param)

        # Convert from log-space to real scale
        log_min = torch.log10(self.mins.to(device)).unsqueeze(0).expand_as(raw_action)
        log_max = torch.log10(self.maxs.to(device)).unsqueeze(0).expand_as(raw_action)
        clamped_log = torch.clamp(raw_action, min=log_min, max=log_max)
        action = 10 ** clamped_log  # Final real action

        # Log probability in log space (no Jacobian adjustment needed since raw_action is detached)
        log_prob = -0.5 * ((raw_action - mu) / std) ** 2 - self.log_std - 0.5 * math.log(2 * math.pi)

        return {
            k: {
                "action": action[:, i],
                "log_prob": log_prob[:, i],
                "log_action": clamped_log[:, i]
            }
            for i, k in enumerate(self.keys)
        }




class PhaseShiftedLogGaussianPolicy(nn.Module):
    def __init__(self, key_info_list, init_std=-2.0, amplitude=0.05, period=20.0):
        super().__init__()
        self.keys = [k for k, _, _, _ in key_info_list]
        self.mins = torch.tensor([min_v for _, _, min_v, _ in key_info_list])
        self.maxs = torch.tensor([max_v for _, _, _, max_v in key_info_list])


        init_mu = torch.tensor([math.log10(init) for _, init, _, _ in key_info_list])
        self.base_mu = nn.Parameter(init_mu)

        self.log_std = nn.Parameter(torch.ones(len(self.keys)) * init_std)


        self.amplitude = amplitude
        self.period = period


        self.register_buffer("phase_shift", torch.rand(len(self.keys)) * 2 * math.pi)

    def forward(self, prev_step_loss=None, step=0, batch_size=1, device=None):
        if device is None:
            device = self.base_mu.device

        if not torch.is_tensor(step):
            step = torch.tensor(step, dtype=torch.float32, device=device)

        mu = self.base_mu.unsqueeze(0).expand(batch_size, -1).clone()


        for i in range(len(self.keys)):
            modulation = self.amplitude * torch.sin(2 * math.pi * step / self.period + self.phase_shift[i])
            mu[:, i] += modulation

        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)

        eps = torch.randn_like(mu)
        raw_action = (mu + eps * std).detach()

        # log-scale clamping
        log_min = torch.log10(self.mins.to(device)).unsqueeze(0).expand_as(raw_action)
        log_max = torch.log10(self.maxs.to(device)).unsqueeze(0).expand_as(raw_action)
        clamped_log = torch.clamp(raw_action, min=log_min, max=log_max)

        action = 10 ** clamped_log


        log_prob = -0.5 * ((raw_action - mu) / std) ** 2 - self.log_std - 0.5 * math.log(2 * math.pi)

        return {
            k: {
                "action": action[:, i],
                "log_prob": log_prob[:, i],
                "log_action": clamped_log[:, i]
            }
            for i, k in enumerate(self.keys)
        }


class SigmoidScaledGaussianPolicy(nn.Module):
    def __init__(self, key_info_list, init_std=-1.0, init_weight=-0.5):
        super().__init__()
        self.keys = [k for k, _, _, _ in key_info_list]
        self.mins = torch.tensor([min_v for _, _, min_v, _ in key_info_list])
        self.maxs = torch.tensor([max_v for _, _, _, max_v in key_info_list])
        init_mu = torch.tensor([init for _, init, _, _ in key_info_list])

        self.base_mu = nn.Parameter(init_mu)
        self.log_std = nn.Parameter(torch.ones(len(self.keys)) * init_std)

        self.loss_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.keys))
        )
        nn.init.zeros_(self.loss_encoder[-1].weight)
        nn.init.zeros_(self.loss_encoder[-1].bias)
        with torch.no_grad():
            self.loss_encoder[-1].bias.copy_(torch.ones(len(self.keys)) * (init_weight))
    

    def forward(self, prev_step_loss, step, batch_size=1, device=None):
        if device is None:
            device = self.base_mu.device

        if not torch.is_tensor(prev_step_loss):
            prev_step_loss = torch.tensor(prev_step_loss, dtype=torch.float32, device=device)
        if not torch.is_tensor(step):
            step = torch.tensor(step, dtype=torch.float32, device=device)

        loss_input = torch.stack((prev_step_loss, 1 - step))
        loss_input = loss_input.view(1, 2).expand(batch_size, -1)  # [B, 2]

        loss_offset = self.loss_encoder(loss_input)  # [B, N]

        mu = self.base_mu.unsqueeze(0).expand(batch_size, -1) + loss_offset  # [B, N]
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)  # [B, N]

        eps = torch.randn_like(mu)
        raw_action = (mu + eps * std).detach() 

        #  
        sigmoid_output = torch.sigmoid(raw_action)

        #  
        min_v = self.mins.to(device).unsqueeze(0).expand_as(sigmoid_output)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(sigmoid_output)
        action = sigmoid_output * (max_v - min_v) + min_v

        #  
        log_prob = -0.5 * ((raw_action - mu)/std) ** 2 - self.log_std - 0.5 * math.log(2 * math.pi)

        return {
            k: {
                "action": action[:, i],
                "log_prob": log_prob[:, i]
            }
            for i, k in enumerate(self.keys)
        }
        


class SigmoidScaledGaussianPolicyStateLess(nn.Module):
    def __init__(self, key_info_list, init_std=-1.0):
        super().__init__()
        self.keys = [k for k, _, _, _ in key_info_list]
        self.mins = torch.tensor([min_v for _, _, min_v, _ in key_info_list])
        self.maxs = torch.tensor([max_v for _, _, _, max_v in key_info_list])
        init_mu = torch.tensor([init for _, init, _, _ in key_info_list])

        self.base_mu = nn.Parameter(init_mu)
        self.log_std = nn.Parameter(torch.ones(len(self.keys)) * init_std)

    def forward(self, batch_size=1, device=None):
        if device is None:
            device = self.base_mu.device

        mu = self.base_mu.unsqueeze(0).expand(batch_size, -1)  # [B, N]
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)  # [B, N]

        eps = torch.randn_like(mu)
        raw_action = (mu + eps * std).detach()   # [B, N]


        sigmoid_output = torch.sigmoid(raw_action)


        min_v = self.mins.to(device).unsqueeze(0).expand_as(sigmoid_output)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(sigmoid_output)
        action = sigmoid_output * (max_v - min_v) + min_v


        log_prob = -0.5 * ((raw_action - mu)/std) ** 2 - self.log_std - 0.5 * math.log(2 * math.pi)

        return {
            k: {
                "action": action[:, i],
                "log_prob": log_prob[:, i]
            }
            for i, k in enumerate(self.keys)
        }
class SigmoidLearnableScaledPolicy(nn.Module):
    def __init__(self, key_info_list, init_std=-1.0):
        super().__init__()
        self.keys = [k for k, _, _, _ in key_info_list]
        self.mins = torch.tensor([min_v for _, _, min_v, _ in key_info_list])
        self.maxs = torch.tensor([max_v for _, _, _, max_v in key_info_list])
        init_mu = torch.tensor([init for _, init, _, _ in key_info_list])

        self.base_mu = nn.Parameter(init_mu)
        self.log_std = nn.Parameter(torch.ones(len(self.keys)) * init_std)


        init_scale = [(max_v - min_v) * 0.1 for _, _, min_v, max_v in key_info_list]
        init_bias = [(max_v + min_v) / 2.0 for _, _, min_v, max_v in key_info_list]
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.bias = nn.Parameter(torch.tensor(init_bias))

        self.loss_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.keys))
        )
        nn.init.zeros_(self.loss_encoder[-1].weight)
        nn.init.zeros_(self.loss_encoder[-1].bias)

    def forward(self, prev_step_loss, step, batch_size=1, device=None):
        if device is None:
            device = self.base_mu.device

        if not torch.is_tensor(prev_step_loss):
            prev_step_loss = torch.tensor(prev_step_loss, dtype=torch.float32, device=device)
        if not torch.is_tensor(step):
            step = torch.tensor(step, dtype=torch.float32, device=device)

        loss_input = torch.stack((prev_step_loss, 1 - step))
        loss_input = loss_input.view(1, 2).expand(batch_size, -1)

        loss_offset = self.loss_encoder(loss_input)

        mu = self.base_mu.unsqueeze(0).expand(batch_size, -1) + loss_offset
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)
        eps = torch.randn_like(mu)
        raw_action = mu + eps * std


        scale = self.scale.unsqueeze(0).expand_as(mu).to(device)
        bias = self.bias.unsqueeze(0).expand_as(mu).to(device)
        action = torch.sigmoid(raw_action) * scale + bias


        min_v = self.mins.to(device).unsqueeze(0).expand_as(action)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(action)
        action = torch.clamp(action, min=min_v, max=max_v)

        log_prob = -0.5 * ((raw_action - mu)/std) **2 - self.log_std - 0.5 * math.log(2 * math.pi)

        return {
            k: {
                "action": action[:, i],
                "log_prob": log_prob[:, i]
            }
            for i, k in enumerate(self.keys)
        }
        
        

class SigmoidLearnableScaledPolicyTime(nn.Module):
    def __init__(self, key_info_list, init_std=-1.0):
        super().__init__()
        self.keys = [k for k, _, _, _ in key_info_list]
        self.mins = torch.tensor([min_v for _, _, min_v, _ in key_info_list])
        self.maxs = torch.tensor([max_v for _, _, _, max_v in key_info_list])
        init_mu = torch.tensor([init for _, init, _, _ in key_info_list])

        self.base_mu = nn.Parameter(init_mu)
        self.log_std = nn.Parameter(torch.ones(len(self.keys)) * init_std)


        init_scale = [(max_v - min_v) * 0.1 for _, _, min_v, max_v in key_info_list]
        init_bias = [(max_v + min_v) / 2.0 for _, _, min_v, max_v in key_info_list]
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.bias = nn.Parameter(torch.tensor(init_bias))

        self.loss_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.keys))
        )
        nn.init.zeros_(self.loss_encoder[-1].weight)
        nn.init.zeros_(self.loss_encoder[-1].bias)

    def forward(self, prev_step_loss, step, batch_size=1, device=None):
        if device is None:
            device = self.base_mu.device

        if not torch.is_tensor(prev_step_loss):
            prev_step_loss = torch.tensor(prev_step_loss, dtype=torch.float32, device=device)
        if not torch.is_tensor(step):
            step = torch.tensor(step, dtype=torch.float32, device=device)
        
        sin_t = torch.sin(2 * math.pi * step)
        cos_t = torch.cos(2 * math.pi * step)
        loss_input = torch.stack((prev_step_loss, 1 - step, sin_t, cos_t))

        #loss_input = torch.stack((prev_step_loss, 1 - step))
        loss_input = loss_input.view(1, 4).expand(batch_size, -1)

        loss_offset = self.loss_encoder(loss_input)

        mu = self.base_mu.unsqueeze(0).expand(batch_size, -1) + loss_offset
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)
        eps = torch.randn_like(mu)
        raw_action = mu + eps * std


        scale = self.scale.unsqueeze(0).expand_as(mu).to(device)
        bias = self.bias.unsqueeze(0).expand_as(mu).to(device)
        action = torch.sigmoid(raw_action) * scale + bias


        min_v = self.mins.to(device).unsqueeze(0).expand_as(action)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(action)
        action = torch.clamp(action, min=min_v, max=max_v)

        log_prob = -0.5 * eps**2 - self.log_std - 0.5 * math.log(2 * math.pi)

        return {
            k: {
                "action": action[:, i],
                "log_prob": log_prob[:, i]
            }
            for i, k in enumerate(self.keys)
        }




class SigmoidLearnableScaledReisdualPolicy(nn.Module):
    def __init__(self, key_info_list, init_std=-1.0, init_weight=-0.5):
        super().__init__()
        self.keys = [k for k, _, _, _ in key_info_list]
        self.mins = torch.tensor([min_v for _, _, min_v, _ in key_info_list])
        self.maxs = torch.tensor([max_v for _, _, _, max_v in key_info_list])
        init_mu = torch.tensor([init for _, init, _, _ in key_info_list])

        self.mu = nn.Parameter(init_mu)
        self.log_std = nn.Parameter(torch.ones(len(self.keys)) * init_std)


        init_scale = [(max_v - min_v) * 0.1 for _, _, min_v, max_v in key_info_list]
        init_bias = [(max_v + min_v) / 2.0 for _, _, min_v, max_v in key_info_list]
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.bias = nn.Parameter(torch.tensor(init_bias))

        self.loss_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.keys))
        )
        nn.init.zeros_(self.loss_encoder[-1].weight)
        nn.init.zeros_(self.loss_encoder[-1].bias)
        
        
        with torch.no_grad():
            self.loss_encoder[-1].bias.copy_(torch.ones(len(self.keys)) * (init_weight))
            

    def forward(self, prev_step_loss, step, last_action=None, batch_size=1, device=None):
        if device is None:
            device = self.mu.device

        if not torch.is_tensor(prev_step_loss):
            prev_step_loss = torch.tensor(prev_step_loss, dtype=torch.float32, device=device)
        if not torch.is_tensor(step):
            step = torch.tensor(step, dtype=torch.float32, device=device)


        if last_action is None:
            last_action = torch.zeros(len(self.keys), dtype=torch.float32, device=device)
        last_action = last_action.view(1, -1).expand(batch_size, -1)

        loss_input = torch.stack((prev_step_loss, 1 - step))
        loss_input = loss_input.view(1, 2).expand(batch_size, -1)

        loss_offset = self.loss_encoder(loss_input)
        mu = self.mu.unsqueeze(0).expand(batch_size, -1) + loss_offset
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)
        eps = torch.randn_like(mu)
        delta_action = (mu + eps * std).detach()


        raw_action = last_action + delta_action


        scale = self.scale.unsqueeze(0).expand_as(raw_action).to(device)
        bias = self.bias.unsqueeze(0).expand_as(raw_action).to(device)
        action = torch.sigmoid(raw_action) * scale + bias


        min_v = self.mins.to(device).unsqueeze(0).expand_as(action)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(action)
        action = torch.clamp(action, min=min_v, max=max_v)

        log_prob = -0.5 * ((delta_action - mu)/std) **2 - self.log_std - 0.5 * math.log(2 * math.pi)

        return {
            k: {
                "action": action[:, i],
                "log_prob": log_prob[:, i]
            }
            for i, k in enumerate(self.keys)
        }
        
        
class SigmoidLearnableScaledPolicyCamEncoding(nn.Module):
    def __init__(self, key_info_list, init_std=-1.0):
        super().__init__()
        self.keys = [k for k, _, _, _ in key_info_list]
        self.mins = torch.tensor([min_v for _, _, min_v, _ in key_info_list])
        self.maxs = torch.tensor([max_v for _, _, _, max_v in key_info_list])
        init_mu = torch.tensor([init for _, init, _, _ in key_info_list])

        self.base_mu = nn.Parameter(init_mu)
        self.log_std = nn.Parameter(torch.ones(len(self.keys)) * init_std)


        init_scale = [(max_v - min_v) * 0.1 for _, _, min_v, max_v in key_info_list]
        init_bias = [(max_v + min_v) / 2.0 for _, _, min_v, max_v in key_info_list]
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.bias = nn.Parameter(torch.tensor(init_bias))


        self.loss_encoder = nn.Sequential(
            nn.Linear(14, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.keys))
        )
        nn.init.zeros_(self.loss_encoder[-1].weight)
        nn.init.zeros_(self.loss_encoder[-1].bias)

    def forward(self, prev_step_loss, step, camera_matrix, batch_size=1, device=None):
        if device is None:
            device = self.base_mu.device


        if not torch.is_tensor(prev_step_loss):
            prev_step_loss = torch.tensor(prev_step_loss, dtype=torch.float32, device=device)
        if not torch.is_tensor(step):
            step = torch.tensor(step, dtype=torch.float32, device=device)

        # 🔥 flatten 3x4 camera matrix → [12]
        cam_feat = camera_matrix[:3, :4].reshape(-1).to(device)


        scalar_inputs = torch.tensor([prev_step_loss.item(), 1 - step.item()], dtype=torch.float32, device=device)
        loss_input = torch.cat([cam_feat, scalar_inputs], dim=0).unsqueeze(0).expand(batch_size, -1)  # [B, 14]


        loss_offset = self.loss_encoder(loss_input)  # [B, N]

        mu = self.base_mu.unsqueeze(0).expand(batch_size, -1) + loss_offset  # [B, N]
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)             # [B, N]
        eps = torch.randn_like(mu)
        raw_action = mu + eps * std


        scale = self.scale.unsqueeze(0).expand_as(mu).to(device)
        bias = self.bias.unsqueeze(0).expand_as(mu).to(device)
        action = torch.sigmoid(raw_action) * scale + bias


        min_v = self.mins.to(device).unsqueeze(0).expand_as(action)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(action)
        action = torch.clamp(action, min=min_v, max=max_v)

        log_prob = -0.5 * eps ** 2 - self.log_std - 0.5 * math.log(2 * math.pi)

        return {
            k: {
                "action": action[:, i],
                "log_prob": log_prob[:, i]
            }
            for i, k in enumerate(self.keys)
        }

class BetaPolicy(nn.Module):
    def __init__(self, key_info_list):
        super().__init__()
        self.keys = [k for k, _, _, _ in key_info_list]
        self.mins = torch.tensor([min_v for _, _, min_v, _ in key_info_list])
        self.maxs = torch.tensor([max_v for _, _, _, max_v in key_info_list])


        self.log_alpha = nn.Parameter(torch.zeros(len(self.keys)))
        self.log_beta = nn.Parameter(torch.zeros(len(self.keys)))

        self.loss_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.keys) * 2)
        )
        nn.init.zeros_(self.loss_encoder[-1].weight)
        nn.init.zeros_(self.loss_encoder[-1].bias)


        self.temperature = nn.Parameter(torch.tensor(0.5))

    def forward(self, prev_step_loss, step, batch_size=1, device=None):
        if device is None:
            device = self.log_alpha.device

        if not torch.is_tensor(prev_step_loss):
            prev_step_loss = torch.tensor(prev_step_loss, dtype=torch.float32, device=device)
        if not torch.is_tensor(step):
            step = torch.tensor(step, dtype=torch.float32, device=device)

        loss_input = torch.stack((prev_step_loss, 1 - step))
        loss_input = loss_input.view(1, 2).expand(batch_size, -1)  # [B, 2]

        offsets = self.loss_encoder(loss_input)  # [B, 2N]
        alpha_offset, beta_offset = torch.chunk(offsets, 2, dim=-1)

        alpha = torch.exp(self.log_alpha.unsqueeze(0) + alpha_offset).to(device) + 1e-3
        beta = torch.exp(self.log_beta.unsqueeze(0) + beta_offset).to(device) + 1e-3

        dist = D.RelaxedBeta(self.temperature, alpha, beta)
        sample = dist.rsample()  # [B, N] ∈ [0,1]


        min_v = self.mins.to(device).unsqueeze(0).expand_as(sample)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(sample)
        action = sample * (max_v - min_v) + min_v

        log_prob = dist.log_prob(sample)

        return {
            k: {
                "action": action[:, i],
                "log_prob": log_prob[:, i]
            }
            for i, k in enumerate(self.keys)
        }
        
        



class SimpleScalarPolicyClampSTDV2(nn.Module):
    """
    Policy 完全由可学习的 6 个 μ 和 6 个 log σ 组成，和环境 state 无关。
    每次调用 forward() 返回一批动作样本及其对数概率。
    """
    def __init__(self, keys, init_std=-1, clamp=False, detach=True, use_schedule=True, total_steps=30000):
        super().__init__()

        self.keys = keys
        self.mu      = nn.Parameter(torch.ones(len(self.keys)))   # shape [6]
        self.log_std = nn.Parameter(torch.ones(len(self.keys)) * init_std)  # shape [6]
        self.clamp = clamp
        self.detach = detach


        self.use_schedule = use_schedule
        self.total_steps = total_steps
        self.current_step = 0
        self.min_log_std = torch.tensor(-2.0).cuda()
        self.max_log_std = torch.tensor(0.5).cuda()

    def forward(self, batch_size: int = 1, device=None):
        """
        如果你真的不需要任何输入，把 forward 设计成接受 batch_size 就行；
        仍然兼容 torchscript / DDP 并且易于和采样器配合。
        """
        if device is None:
            device = self.mu.device


        if self.use_schedule:
            log_std_value = self.cosine_log_std_schedule(self.current_step, self.total_steps)
            self.log_std.data.fill_(log_std_value)
            self.current_step += 1

        mu = self.mu.unsqueeze(0).expand(batch_size, -1).to(device)       # [B,6]
        if self.clamp:
            log_std = torch.clamp(self.log_std, -20, 1)
            std = torch.exp(log_std).unsqueeze(0).expand_as(mu).to(device)
        else:
            std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu).to(device)  # [B,6]

        eps = torch.randn_like(mu)          # [B,6]
        if self.detach:
            actions = (mu + eps * std).detach()                # re‑parameterization trick
        else:
            actions = (mu + eps * std)


        log_probs = -0.5 * (((actions - mu) / std) ** 2) - self.log_std - 0.5 * math.log(2 * math.pi)  # [B,6]


        out = {
            k: {
                "action": actions[:, i],        # shape [B]
                "log_prob": log_probs[:, i]     # shape [B]
            }
            for i, k in enumerate(self.keys)
        }
        return out

    def cosine_log_std_schedule(self, step, total_steps):
        """
        Cosine schedule: 头尾小，中间大。
        """
        #progress = step / total_steps

        t = torch.tensor(step / total_steps, device=self.log_std.device)


        coeff = torch.sin(math.pi * t).pow(2)


        value = self.min_log_std + (self.max_log_std - self.min_log_std) * coeff
        return value
    
    



class LogNormalLRPolicyState(nn.Module):
    """
    输入一个 state 向量，输出
        μ_z(s) ∈ ℝ^6,   logσ_z(s) ∈ ℝ^6
    然后按 log-normal 采样 factor = exp(z) (>0)
    """

    def __init__(self, keys, state_dim,
                 hid_dim=64,
                 init_std=-2.0, clamp=False, detach=True):
        super().__init__()
        self.keys = keys
        self.K = len(keys)
        self.clamp = clamp
        self.detach = detach


        self.mu_net = nn.Sequential(
            nn.Linear(state_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, self.K)
        )

        self.logstd_net = nn.Sequential(
            nn.Linear(state_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, self.K),
        )

        nn.init.constant_(self.logstd_net[-1].bias, init_std)

    def forward(self, state, device=None):
        """
        state: Tensor [B, state_dim]
        返回 dict[k] = {"action": factor, "log_prob": logπ}
        """
        if device is None:
            device = state.device
        state = state.to(device)


        mu_z       = self.mu_net(state)                 # [B, K]
        log_std_z  = self.logstd_net(state)             # [B, K]
        if self.clamp:
            log_std_z = torch.clamp(log_std_z, -10, -1)
        std_z      = torch.exp(log_std_z)               # [B, K]


        eps  = torch.randn_like(mu_z)
        z    = mu_z + eps * std_z                       # [B, K]
        if self.detach:
            z = z.detach()
        factor = torch.exp(z)                           # >0

        # -------- 3. log π(z|μ,σ) ----------
        log_prob = -0.5 * ((z - mu_z) / std_z) ** 2\
                   - log_std_z - 0.5 * math.log(2*math.pi) - z  # [B, K]

        out = {k: {"action": factor[:, i],
                   "log_prob": log_prob[:, i]}
               for i, k in enumerate(self.keys)}
        return out
    
    
    


class PerViewScalarPolicyClampSTD(nn.Module):
    """
    Policy 由 [num_views × num_keys] 的 μ 和 log σ 构成，
    forward时根据给定的 view_idx（int 或长整型Tensor）来取对应那组参数。
    """
    def __init__(
        self,
        key_info_list,
        num_views: int,
        init_std: float = 0.0,
        clamp: bool = False,
        detach: bool = True
    ):
        super().__init__()
        self.keys = [name for name, _, _, _ in key_info_list]
        self.num_keys = len(self.keys)
        self.num_views = num_views


        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.register_buffer('mins', torch.tensor(mins))   # [K]
        self.register_buffer('maxs', torch.tensor(maxs))   # [K]


        # shape: [V, K]
        self.mu      = nn.Parameter(torch.ones(num_views, self.num_keys))
        self.log_std = nn.Parameter(torch.ones(num_views, self.num_keys) * init_std)
        self.init_std = init_std

        self.clamp  = clamp
        self.detach = detach

    def forward(self, view_idx, batch_size: int = 1, device=None):
        """
        view_idx: int           -> 同一个 view，返回 [B,K]  
                  LongTensor[B] -> 每个样本可以属于不同 view，返回 [B,K]
        """
        if device is None:
            device = self.mu.device




        mu_v      = self.mu[view_idx].to(device)
        logstd_v  = self.log_std[view_idx].to(device)


        if self.clamp:
            logstd_v = torch.clamp(logstd_v, -20, 1)

        std_v = torch.exp(logstd_v)



        mu_v  = mu_v.unsqueeze(0).expand(batch_size, -1)   if mu_v.dim()==1  else mu_v
        std_v = std_v.unsqueeze(0).expand_as(mu_v)        if std_v.dim()==1 else std_v
        logstd_v = logstd_v.unsqueeze(0).expand_as(mu_v)  if logstd_v.dim()==1 else logstd_v


        eps = torch.randn_like(mu_v)
        raw = mu_v + eps * std_v
        if self.detach:
            raw = raw.detach()


        # log N(mu, sigma^2)
        logp = -0.5 * (((raw - mu_v) / std_v) ** 2)\
               - logstd_v\
               - 0.5 * math.log(2 * math.pi)  # [B,K]


        min_v = self.mins.to(device).unsqueeze(0).expand_as(raw)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(raw)
        action = torch.clamp(raw, min=min_v, max=max_v)




        out = {
            k: {
                "action":    action[:, i],   # [B]
                "log_prob":  logp[:, i]       # [B]
            }
            for i, k in enumerate(self.keys)
        }
        return out
    def reset_default_weights(self):
        """
        Reset μ back to 1.0 and log_std back to the original init_std across all views.
        """
        with torch.no_grad():
            # μ = 1 for every view/key
            self.mu.data.fill_(1.0)
            # log_std = init_std for every view/key
            self.log_std.data.fill_(self.init_std)



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PerViewMixturePolicy(nn.Module):
    """
    Policy 由 [num_views × num_keys × num_components] 的 logits, μ 和 log σ 构成，
    forward 时根据给定的 view_idx 取对应那组参数，生成高斯混合采样与 log 概率。
    """
    def __init__(
        self,
        key_info_list,
        num_views: int,
        num_components: int = 5,
        init_std: float = 0.0,
        clamp: bool = False,
        detach: bool = True
    ):
        super().__init__()
        self.keys = [name for name, _, _, _ in key_info_list]
        self.num_keys = len(self.keys)
        self.num_views = num_views
        self.num_components = num_components


        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.register_buffer('mins', torch.tensor(mins))   # [K]
        self.register_buffer('maxs', torch.tensor(maxs))   # [K]


        # shape: [V, K, C]
        self.logits   = nn.Parameter(torch.zeros(num_views, self.num_keys, num_components))
        self.mu       = nn.Parameter(torch.zeros(num_views, self.num_keys, num_components))
        self.log_std  = nn.Parameter(torch.ones(num_views, self.num_keys, num_components) * init_std)

        self.clamp    = clamp
        self.detach   = detach

    def forward(self, view_idx, batch_size: int = 1, device=None):
        """
        view_idx: int           -> 同一个 view，返回 [B,K]  
                  LongTensor[B] -> 每个样本可以属于不同 view，返回 [B,K]
        """
        if device is None:
            device = self.mu.device



        logits_v   = self.logits[view_idx].to(device)
        mu_v       = self.mu[view_idx].to(device)
        logstd_v   = self.log_std[view_idx].to(device)


        if self.clamp:
            logstd_v = torch.clamp(logstd_v, -20, 1)
        std_v = torch.exp(logstd_v)  # [K,C] or [B,K,C]


        if mu_v.dim() == 2:
            # [K,C] -> [B,K,C]
            mu_v     = mu_v.unsqueeze(0).expand(batch_size, -1, -1)
            std_v    = std_v.unsqueeze(0).expand(batch_size, -1, -1)
            logstd_v = logstd_v.unsqueeze(0).expand(batch_size, -1, -1)
            logits_v = logits_v.unsqueeze(0).expand(batch_size, -1, -1)


        weights = F.softmax(logits_v, dim=-1)          # [B,K,C]
        log_weights = F.log_softmax(logits_v, dim=-1)  # [B,K,C]



        flat_w = weights.view(-1, self.num_components)
        comp_idx = torch.multinomial(flat_w, num_samples=1).squeeze(-1)  # [B*K]
        comp_idx = comp_idx.view(batch_size, self.num_keys)             # [B,K]



        idx = comp_idx.unsqueeze(-1)  # [B,K,1]
        mu_sel     = torch.gather(mu_v,     dim=2, index=idx).squeeze(-1)  # [B,K]
        std_sel    = torch.gather(std_v,    dim=2, index=idx).squeeze(-1)  # [B,K]
        logstd_sel = torch.gather(logstd_v, dim=2, index=idx).squeeze(-1)  # [B,K]

        eps = torch.randn_like(mu_sel)
        raw = mu_sel + eps * std_sel
        if self.detach:
            raw = raw.detach()



        # raw -> [B,K] -> expand to [B,K,C]
        raw_exp = raw.unsqueeze(-1).expand_as(mu_v)
        comp_logp = -0.5 * (((raw_exp - mu_v) / std_v) ** 2)\
                    - logstd_v\
                    - 0.5 * math.log(2 * math.pi)  # [B,K,C]

        mix_logp = torch.logsumexp(log_weights + comp_logp, dim=-1)  # [B,K]


        min_v = self.mins.to(device).unsqueeze(0).expand_as(raw)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(raw)
        action = torch.clamp(raw, min=min_v, max=max_v)  # [B,K]


        out = {
            k: {
                "action":   action[:, i],   # [B]
                "log_prob": mix_logp[:, i]  # [B]
            }
            for i, k in enumerate(self.keys)
        }
        return out




class ProjectedGaussianPolicy(nn.Module):
    """
    根据每个 view 对应的 projection matrix 动态生成 (μ, σ)，
    forward 时根据 view_idx 取出该 view 的 projection matrix，
    用两条线性头分别预测 μ 和 logσ，再做 reparameterization 采样，
    并计算 N(μ,σ^2) 下的 log-prob，最后 clamp 到动作边界。
    """
    def __init__(
        self,
        key_info_list,
        proj_matrices: torch.Tensor,
        init_log_std: float = 0.0,
        clamp: bool = False,
        detach: bool = True
    ):
        super().__init__()

        self.keys      = [name for name, _, _, _ in key_info_list]
        self.num_keys  = len(self.keys)


        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.register_buffer('mins', torch.tensor(mins))
        self.register_buffer('maxs', torch.tensor(maxs))



        V = proj_matrices.size(0)
        flat = proj_matrices.view(V, -1)
        self.register_buffer('proj_mat_flat', flat)  # [V, F]
        feat_dim = flat.size(1)
        self.num_views = V


        self.mu_head      = nn.Linear(feat_dim, self.num_keys)
        self.logstd_head = nn.Linear(feat_dim, self.num_keys)


        nn.init.zeros_(self.logstd_head.weight)

        nn.init.constant_(self.logstd_head.bias, init_log_std)
         
        # 

        self.clamp  = clamp
        self.detach = detach

    def forward(self, view_idx, batch_size: int = 1, device=None):
        """
        Args:
          view_idx: int 或 LongTensor[B]
          batch_size: 当 view_idx 是 int 时生效
        Returns:
          { key_i: {"action": Tensor[B], "log_prob": Tensor[B]} }
        """
        if device is None:
            device = self.proj_mat_flat.device


        if torch.is_tensor(view_idx):
            # view_idx: [B]
            B = view_idx.size(0)
            # [B, F]
            feat = self.proj_mat_flat[view_idx].to(device)
        else:

            B = batch_size
            # [F] -> expand [B, F]
            feat = self.proj_mat_flat[view_idx].to(device).unsqueeze(0).expand(B, -1)


        mu     = self.mu_head(feat)
        logstd = self.logstd_head(feat)

        # 2) clamp logσ
        if self.clamp:
            logstd = torch.clamp(logstd, -20, 1)
        std = torch.exp(logstd)  # [B, K]


        eps = torch.randn_like(mu)
        raw = mu + eps * std
        if self.detach:
            raw = raw.detach()


        logp = (
            -0.5 * (((raw - mu) / std) ** 2)
            - logstd
            - 0.5 * math.log(2 * math.pi)
        )


        min_v = self.mins.to(device).unsqueeze(0).expand(B, -1)
        max_v = self.maxs.to(device).unsqueeze(0).expand(B, -1)
        action = torch.clamp(raw, min=min_v, max=max_v)


        return {
            key: {
                "action":   action[:, i],   # [B]
                "log_prob": logp[:, i]      # [B]
            }
            for i, key in enumerate(self.keys)
        }
        
        


class LatentMLPPolicyClampSTDClamp(nn.Module):
    """
    Policy 的参数来自一个可学习潜在向量 z 经由 MLP 产生 μ 和 log σ，
    与环境 state 无关。forward 只需 batch_size（和 device）。
    每次调用返回一批动作样本及其对数概率。
    """
    def __init__(
        self,
        key_info_list,
        latent_dim: int = 32,
        hidden_sizes=(64, 64),
        init_std: float = 0.1,
        clamp: bool = False,
        detach: bool = True,
    ):
        super().__init__()
        # action bounds
        self.keys = [name for name, _, _, _ in key_info_list]
        self.action_dim = len(self.keys)
        self.mins = torch.tensor([min_v / default for _, default, min_v, _ in key_info_list])
        self.maxs = torch.tensor([max_v / default for _, default, _, max_v in key_info_list])


        self.z = nn.Parameter(torch.randn(latent_dim) * init_std)

        # MLP body：latent_dim -> ... -> hidden -> 2*action_dim
        layers = []
        prev = latent_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 2 * self.action_dim))
        self.net = nn.Sequential(*layers)

        self.clamp_std = clamp 
        self.detach_action = detach
        
        

        last_linear = self.net[-1]  # type: nn.Linear

        nn.init.zeros_(last_linear.weight)

        nn.init.constant_(last_linear.bias[:self.action_dim], 1.0)

        nn.init.constant_(last_linear.bias[self.action_dim:], -1.0)

    def forward(self, batch_size: int = 1, device=None):
        if device is None:
            device = self.z.device

        # [B, latent_dim]
        z_batch = self.z.unsqueeze(0).expand(batch_size, -1).to(device)
        # [B, 2*action_dim]
        out = self.net(z_batch)

        mu, log_std = out[:, :self.action_dim], out[:, self.action_dim:]

        if self.clamp_std:
            log_std = torch.clamp(log_std, -20, 1)
        std = torch.exp(log_std)  # [B, action_dim]


        eps = torch.randn_like(mu)
        raw_action = mu + eps * std
        if self.detach_action:
            raw_action = raw_action.detach()


        log_probs = -0.5 * (((raw_action - mu) / std) ** 2)\
                    - log_std - 0.5 * math.log(2 * math.pi)  # [B, action_dim]


        min_v = self.mins.to(device).unsqueeze(0).expand_as(raw_action)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(raw_action)
        action = torch.clamp(raw_action, min=min_v, max=max_v)


        return {
            k: {
                "action": action[:, i],      # shape [1]
                "log_prob": log_probs[:, i]  # shape [1]
            }
            for i, k in enumerate(self.keys)
        }

        
        


class PoseMLPPolicyClampSTDClamp(nn.Module):
    """
    Policy 由两个 4×4 pose 矩阵作为输入，通过 MLP 产生 μ 和 log σ，
    与环境 state 无关，只依赖这两个 pose。
    """
    def __init__(
        self,
        key_info_list,
        hidden_sizes=(64, 64),
        clamp_std: bool = False,
        detach_action: bool = True,
    ):
        super().__init__()

        self.keys = [name for name, _, _, _ in key_info_list]
        self.action_dim = len(self.keys)
        self.mins = torch.tensor([min_v / default for _, default, min_v, _ in key_info_list])
        self.maxs = torch.tensor([max_v / default for _, default, _, max_v in key_info_list])


        input_dim = 4 * 4 * 2


        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 2 * self.action_dim))
        self.net = nn.Sequential(*layers)


        last = self.net[-1]  # type: nn.Linear
        nn.init.zeros_(last.weight)

        nn.init.constant_(last.bias[:self.action_dim],  1.0)

        nn.init.constant_(last.bias[self.action_dim:], -1.0)

        self.clamp_std     = clamp_std
        self.detach_action = detach_action

    def forward(self, pose1: torch.Tensor, pose2: torch.Tensor):
        """
        输入
            pose1, pose2: Tensor [B,4,4]
        返回
            B==1 时：dict[k] -> {"action": Tensor[1], "log_prob": Tensor[1]}
            B>1  时：list 长度 B，每项同上
        """
        B = pose1.shape[0]
        device = pose1.device

        # flatten + cat -> [B,32]
        p1 = pose1.view(B, -1)
        p2 = pose2.view(B, -1)
        x  = torch.cat([p1, p2], dim=1).to(device)


        out = self.net(x)
        mu      = out[:, :self.action_dim]            # [B, D]
        log_std = out[:, self.action_dim:]            # [B, D]
        if self.clamp_std:
            log_std = torch.clamp(log_std, -20, 1)
        std = torch.exp(log_std)                      # [B, D]

        # re-parameterize
        eps = torch.randn_like(mu)
        raw_action = mu + eps * std
        if self.detach_action:
            raw_action = raw_action.detach()

        # log π(a|·)
        log_probs = -0.5 * (((raw_action - mu) / std) ** 2)\
                    - log_std - 0.5 * math.log(2 * math.pi)  # [B, D]


        min_v = self.mins.to(device).unsqueeze(0).expand_as(raw_action)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(raw_action)
        action = torch.clamp(raw_action, min=min_v, max=max_v)


        def pack_row(a_row, lp_row):
            return {
                k: {
                    "action": a_row[:, i],      # shape [1]
                    "log_prob": lp_row[:, i],   # shape [1]
                }
                for i, k in enumerate(self.keys)
            }

        if B == 1:
            return pack_row(action, log_probs)
        else:
            return [pack_row(action[b:b+1], log_probs[b:b+1]) for b in range(B)]
        
        
        
        
    
    


class PerViewScalarPolicyWithLossState(nn.Module):
    """
    在 PerViewScalarPolicyClampSTD 基础上：
      - 根据 step 线性收紧 logσ；
      - 用 EMA 平滑 loss，并用 loss_norm 调整 μ 和 logσ；
      - 其余逻辑与原版一致（detach、clamp、动作边界、重参数化采样）。
    """
    def __init__(
        self,
        key_info_list,
        num_views: int,
        init_std: float = 0.0,
        clamp: bool = True,
        detach: bool = True,
        max_steps: int = 10000,
        loss_ema_alpha: float = 0.5,
        ema_scale: float = 5.0,
        end_max: float = 5.0
    ):
        super().__init__()

        self.keys = [name for name, _, _, _ in key_info_list]
        self.num_keys = len(self.keys)
        self.num_views = num_views


        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.register_buffer('mins',  torch.tensor(mins))  # [K]
        self.register_buffer('maxs',  torch.tensor(maxs))  # [K]


        self.mu      = nn.Parameter(torch.ones(num_views, self.num_keys))
        self.log_std = nn.Parameter(torch.ones(num_views, self.num_keys) * init_std)
        self.init_std = init_std


        self.clamp     = clamp
        self.detach    = detach
        self.max_steps = max_steps * 2


        self.loss_ema_alpha = loss_ema_alpha
        self.ema_scale      = ema_scale

        self.register_buffer('ema_loss',        torch.zeros(1))
        self.register_buffer('ema_initialized', torch.zeros(1, dtype=torch.bool))


        self.register_buffer('step_buffer', torch.zeros(1, dtype=torch.long))



        self.w_mu     = nn.Parameter(torch.zeros(self.num_keys))
        self.w_logstd = nn.Parameter(torch.zeros(self.num_keys))
        self.end_max = end_max

    def forward(
        self,
        view_idx,
        batch_size: int = 1,
        device=None,
        loss: torch.Tensor = None,
        step: int = None,
    ):
        if device is None:
            device = self.mu.device


        if step is not None:
            self.step_buffer.copy_(torch.tensor(step, dtype=torch.long, device=device))


        if loss is not None:

            loss_val = loss.detach().mean()
            if not self.ema_initialized:

                self.ema_loss[:]        = loss_val
                self.ema_initialized[:] = torch.tensor(True, device=device)
            else:

                self.ema_loss[:] = (
                    self.loss_ema_alpha * self.ema_loss +
                    (1 - self.loss_ema_alpha) * loss_val
                )
            # centered & scale → sigmoid
            centered = loss_val - self.ema_loss
            loss_norm = torch.sigmoid(centered * self.ema_scale)
        else:
            loss_norm = torch.tensor(0.0, device=device)


        mu_v     = self.mu[view_idx].to(device)      # [K] or [B,K]
        logstd_v = self.log_std[view_idx].to(device) # [K] or [B,K]


        w_mu  = self.w_mu.to(device)
        w_ls  = self.w_logstd.to(device)
        mu_v     = mu_v     + loss_norm * w_mu
        logstd_v = logstd_v + loss_norm * w_ls


        std_v = torch.exp(logstd_v)

        if mu_v.dim() == 1:
            mu_v     = mu_v.unsqueeze(0).expand(batch_size, -1)
            std_v    = std_v.unsqueeze(0).expand(batch_size, -1)
            logstd_v = logstd_v.unsqueeze(0).expand(batch_size, -1)

        eps = torch.randn_like(mu_v)
        raw = mu_v + eps * std_v
        if self.detach:
            raw = raw.detach()
            
        
        
        logp = (
            -0.5 * (((raw - mu_v) / std_v) ** 2)
            - logstd_v
            - 0.5 * math.log(2 * math.pi)
        )


        # H = ½(1 + ln(2π)) + ln σ
        const = 0.5 * (1.0 + math.log(2.0 * math.pi))
        entropy_v = logstd_v + const  # [B, K]


        min_v = self.mins.to(device).unsqueeze(0).expand_as(raw)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(raw)
        action = torch.clamp(raw, min_v, max_v)


        return {
            k: {
                "action":   action[:, i],   # [B]
                "log_prob": logp[:, i],     # [B]
                "entropy":  entropy_v[:, i] # [B]
            }
            for i, k in enumerate(self.keys)
        }
        




import math, torch
import torch.nn as nn
from torch import Tensor


# class ScalarPolicyWithSchedState(nn.Module):
#     """



#     """

#     def __init__(
#         self,
#         key_info_list,
#         init_std: float = 0.0,
#         clamp: bool = True,
#         detach: bool = True,


#     ):
#         super().__init__()


#         self.keys      = [name for name, *_ in key_info_list]
#         self.num_keys  = len(self.keys)

#         mins = [min_v / default for _, default, min_v, _ in key_info_list]
#         maxs = [max_v / default for _, default, _, max_v in key_info_list]
#         self.register_buffer("mins", torch.as_tensor(mins))   # [K]
#         self.register_buffer("maxs", torch.as_tensor(maxs))   # [K]


#         self.mu      = nn.Parameter(torch.zeros(self.num_keys))
#         self.log_std = nn.Parameter(torch.ones (self.num_keys) * init_std)



#         self.w_mu_loss,   self.w_mu_step   = nn.Parameter(torch.zeros(self.num_keys)), nn.Parameter(torch.zeros(self.num_keys))
#         self.w_ls_loss,   self.w_ls_step   = nn.Parameter(torch.zeros(self.num_keys)), nn.Parameter(torch.zeros(self.num_keys))


#         self.clamp   = clamp
#         self.detach  = detach
#         self.max_steps  = float(max_steps)


#     # ------------------------------------------------------------------
#     # forward(batch_state)：



#     # ------------------------------------------------------------------
#     def forward(
#         self,
#         *,

#         loss:   Tensor | float | None,    # idem
#         batch_size: int = 1,
#         device=None,
#     ):
#         if device is None:
#             device = self.mu.device

#         step_t = torch.as_tensor(step,  device=device, dtype=torch.float32).view(-1)  \
#                  if step is not None else torch.zeros(batch_size, device=device)
#         loss_t = torch.as_tensor(loss,  device=device, dtype=torch.float32).view(-1)  \
#                  if loss is not None else torch.zeros(batch_size, device=device)

#         if step_t.numel() == 1:   step_t = step_t.expand(batch_size)
#         if loss_t.numel() == 1:   loss_t = loss_t.expand(batch_size)


#         step_norm = (step_t / self.max_steps).clamp_(0., 1.)           # ∈[0,1]
#         loss_norm = torch.sigmoid(loss_t * self.ema_scale)             # ∈(0,1)

#         # ---------- μ & logσ ----------
#         mu      = self.mu + loss_norm.unsqueeze(1) * self.w_mu_loss \
#                         + step_norm.unsqueeze(1) * self.w_mu_step

#         log_std = self.log_std + loss_norm.unsqueeze(1) * self.w_ls_loss \
#                               + step_norm.unsqueeze(1) * self.w_ls_step

#         std  = torch.exp(log_std)


#         eps   = torch.randn(batch_size, self.num_keys, device=device)
#         raw   = mu + eps * std
#         if self.detach:
#             raw = raw.detach()


#         action = torch.clamp(raw,
#                              self.mins.to(device).unsqueeze(0),
#                              self.maxs.to(device).unsqueeze(0))

#         # ---------- log-prob + entropy ----------

#                 - log_std
#                 - 0.5 * math.log(2 * math.pi))

#         entropy = log_std + 0.5 * (1. + math.log(2 * math.pi))

#         return {
#             k: {
#                 "action":   action[:, i],    # [B]
#                 "log_prob": logp  [:, i],
#                 "entropy":  entropy[:, i],
#             }
#             for i, k in enumerate(self.keys)
#         }



# self.mlp = nn.Sequential(
#             nn.Linear(mlp_input,   hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim,  hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2 * self.num_keys)
#         )



class RandomPolicy(nn.Module):
    def __init__(self, key_info_list):
        super().__init__()

        keys = [name for name, *_ in key_info_list]
        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.keys = keys
        self.register_buffer("mins", torch.as_tensor(mins))   # [K]
        self.register_buffer("maxs", torch.as_tensor(maxs))   # [K]

    def forward(self, batch_size: int = 1, device=None):
        if device is None:
            device = self.mins.device


        u = torch.rand(batch_size, len(self.keys), device=device)
        action = self.mins.unsqueeze(0) + u * (self.maxs - self.mins).unsqueeze(0)


        return {
            k: {
                "action":   action[:, i],    # [B]
                "log_prob": torch.full((batch_size,), -torch.log(self.maxs[i]-self.mins[i]), device=device),
                "entropy":  torch.full((batch_size,), torch.log(self.maxs[i]-self.mins[i]), device=device),
            }
            for i, k in enumerate(self.keys)
        }
        
import math, torch
import torch.nn as nn
from torch import Tensor

class ScalarPolicyWithSchedStateNN(nn.Module):
    """
    
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


        self.keys      = [name for name, *_ in key_info_list]
        self.num_keys  = len(self.keys)

        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.register_buffer("mins", torch.as_tensor(mins))   # [K]
        self.register_buffer("maxs", torch.as_tensor(maxs))   # [K]


        self.base_mu      = nn.Parameter(torch.ones(self.num_keys))
        self.base_log_std = nn.Parameter(torch.ones(self.num_keys) * init_std)

        
        # worked mlp3
        self.state_net = nn.Sequential(
            nn.Linear(2,   hidden*2),
            nn.ReLU(),
            nn.Linear(hidden*2,  hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * self.num_keys)
        )
        


        self.clamp   = clamp
        self.detach  = detach
        self.max_steps  = float(max_steps)
        self.ema_scale  = float(ema_scale)
        
        nn.init.zeros_(self.state_net[-1].weight)
        nn.init.zeros_(self.state_net[-1].bias)


    # ------------------------------------------------------------------
    def forward(
        self,
        *,
        step:   Tensor | int | None,
        loss:   Tensor | float | None,
        batch_size: int = 1,
        device=None,
    ):
        if device is None:
            device = self.base_mu.device

        # ---------- scalar → Tensor[B] ----------
        step_t = torch.as_tensor(step,  device=device, dtype=torch.float32).view(-1)\
                 if step is not None else torch.zeros(batch_size, device=device)
        loss_t = torch.as_tensor(loss,  device=device, dtype=torch.float32).view(-1)\
                 if loss is not None else torch.zeros(batch_size, device=device)

        if step_t.numel() == 1:   step_t = step_t.expand(batch_size)
        if loss_t.numel() == 1:   loss_t = loss_t.expand(batch_size)


        step_norm = (step_t / self.max_steps).clamp_(0., 1.)          # ∈[0,1]

        loss_norm = torch.tanh(torch.log1p(loss_t) * self.ema_scale)  # ∈(-1,1)


        state_in  = torch.stack([step_norm, loss_norm], dim=-1)       # [B, 2]
        delta     = self.state_net(state_in)                          # [B, 2K]
        d_mu, d_ls = delta.split(self.num_keys, dim=-1)

        mu       = self.base_mu      + d_mu
        log_std  = self.base_log_std + d_ls

        if self.clamp:

            log_std = torch.clamp(log_std, -6.0, 2.0)

        std = torch.exp(log_std)


        eps  = torch.randn(batch_size, self.num_keys, device=device)
        raw  = mu + eps * std
        if self.detach:
            raw = raw.detach()


        action = torch.clamp(raw,
                             self.mins.to(device).unsqueeze(0),
                             self.maxs.to(device).unsqueeze(0))

        # ---------- log-prob + entropy ----------
        #logp = (-0.5 * eps.pow(2) - log_std - 0.5 * math.log(2 * math.pi))
        
        logp = (
            -0.5 * (((raw - mu) / std) ** 2)
            - log_std
            - 0.5 * math.log(2 * math.pi)
        )
         
        entropy = log_std + 0.5 * (1. + math.log(2 * math.pi))

        return {
            k: {
                "action":   action[:, i],    # [B]
                "log_prob": logp  [:, i],
                "entropy":  entropy[:, i],
            }
            for i, k in enumerate(self.keys)
        }
    
    def reset_parameters(self):
        """
        Re‐initialize all weights / biases exactly as in __init__:
          • base_mu → ones
          • base_log_std → init_std
          • every Linear in state_net → He‐init (except last, which is zeroed)
        """
        # 1) Re‐init the base_mu / base_log_std
        with torch.no_grad():
            self.base_mu.data.fill_(1.0)
            self.base_log_std.data.fill_(self.init_std)

        # 2) Walk through state_net and reinitialize all Linear layers
        #    We’ll use the same logic you’d want: He‐uniform for all linears except
        #    the last one, which you want zeroed (weight & bias = 0).
        #    We know `self.state_net[-1]` is the final Linear, so handle it specially.

        # First, re‐zero the final layer exactly as before:
        last_lin: nn.Linear = self.state_net[-1]
        nn.init.zeros_(last_lin.weight)
        nn.init.zeros_(last_lin.bias)

        # Now re‐init every other Linear in state_net
        # (i.e. indices 0 and 2 in your Sequential:  Linear(2,hidden*2) and Linear(hidden*2,hidden) )
        for module in self.state_net[:-1]:
            if isinstance(module, nn.Linear):
                # Kaiming uniform as a reasonable default (fan_in mode, nonlinearity='relu'):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                # For bias, a common choice is zero (or use the default for Kaiming):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)



class ScalarPolicyWithSchedStateGRUV2(nn.Module):
    """
    
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


        self.keys      = [name for name, *_ in key_info_list]
        self.num_keys  = len(self.keys)

        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.register_buffer("mins", torch.as_tensor(mins))   # [K]
        self.register_buffer("maxs", torch.as_tensor(maxs))   # [K]


        self.base_mu      = nn.Parameter(torch.ones(self.num_keys))
        self.base_log_std = nn.Parameter(torch.ones(self.num_keys) * init_std)
        # )
        self.input_proj = nn.Linear(2, hidden) 
        
        self.state_net = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, 2 * self.num_keys)


        self.clamp   = clamp
        self.detach  = detach
        self.max_steps  = float(max_steps)
        self.ema_scale  = float(ema_scale)
        
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.hidden = None 
        
        for name, param in self.state_net.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    def forward(
        self,
        *,
        step:   Tensor | int | None,
        loss:   Tensor | float | None,
        batch_size: int = 1,
        device=None,
    ):
        if device is None:
            device = self.base_mu.device

        # ---------- scalar → Tensor[B] ----------
        step_t = torch.as_tensor(step,  device=device, dtype=torch.float32).view(-1)\
                 if step is not None else torch.zeros(batch_size, device=device)
        loss_t = torch.as_tensor(loss,  device=device, dtype=torch.float32).view(-1)\
                 if loss is not None else torch.zeros(batch_size, device=device)

        if step_t.numel() == 1:   step_t = step_t.expand(batch_size)
        if loss_t.numel() == 1:   loss_t = loss_t.expand(batch_size)


        step_norm = (step_t / self.max_steps).clamp_(0., 1.)          # ∈[0,1]

        loss_norm = torch.tanh(torch.log1p(loss_t) * self.ema_scale)  # ∈(-1,1)


        state_in  = torch.stack([step_norm, loss_norm], dim=-1)       # [B, 2]
        #delta     = self.state_net(state_in)                          # [B, 2K]
        state_in = self.input_proj(state_in)
        h_prev = self.hidden
        h_new = self.state_net(state_in, h_prev)          # [B,hidden]
        delta = self.out(h_new)                     # [B,2K]
        
        d_mu, d_ls = delta.split(self.num_keys, dim=-1)

        mu       = self.base_mu      + d_mu
        log_std  = self.base_log_std + d_ls

        if self.clamp:

            log_std = torch.clamp(log_std, -6.0, 2.0)

        std = torch.exp(log_std)


        eps  = torch.randn(batch_size, self.num_keys, device=device)
        raw  = mu + eps * std
        if self.detach:
            raw = raw.detach()


        action = torch.clamp(raw,
                             self.mins.to(device).unsqueeze(0),
                             self.maxs.to(device).unsqueeze(0))

        # ---------- log-prob + entropy ----------
        #logp = (-0.5 * eps.pow(2) - log_std - 0.5 * math.log(2 * math.pi))
        
        logp = (
            -0.5 * (((raw - mu) / std) ** 2)
            - log_std
            - 0.5 * math.log(2 * math.pi)
        )
         
        entropy = log_std + 0.5 * (1. + math.log(2 * math.pi))
        self.hidden = h_new.detach()  
        return {
            k: {
                "action":   action[:, i],    # [B]
                "log_prob": logp  [:, i],
                "entropy":  entropy[:, i],
            }
            for i, k in enumerate(self.keys)
        }
    
    def reset_parameters(self):
        """
        Re‐initialize all weights / biases exactly as in __init__:
          • base_mu → ones
          • base_log_std → init_std
          • every Linear in state_net → He‐init (except last, which is zeroed)
        """
        # 1) Re‐init the base_mu / base_log_std
        with torch.no_grad():
            self.base_mu.data.fill_(1.0)
            self.base_log_std.data.fill_(self.init_std)

        # 2) Walk through state_net and reinitialize all Linear layers
        #    We’ll use the same logic you’d want: He‐uniform for all linears except
        #    the last one, which you want zeroed (weight & bias = 0).
        #    We know `self.state_net[-1]` is the final Linear, so handle it specially.

        # First, re‐zero the final layer exactly as before:
        last_lin: nn.Linear = self.state_net[-1]
        nn.init.zeros_(last_lin.weight)
        nn.init.zeros_(last_lin.bias)

        # Now re‐init every other Linear in state_net
        # (i.e. indices 0 and 2 in your Sequential:  Linear(2,hidden*2) and Linear(hidden*2,hidden) )
        for module in self.state_net[:-1]:
            if isinstance(module, nn.Linear):
                # Kaiming uniform as a reasonable default (fan_in mode, nonlinearity='relu'):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                # For bias, a common choice is zero (or use the default for Kaiming):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class ScalarPolicyWithSchedStateGRU(nn.Module):
    """
    
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


        self.keys      = [name for name, *_ in key_info_list]
        self.num_keys  = len(self.keys)

        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.register_buffer("mins", torch.as_tensor(mins))   # [K]
        self.register_buffer("maxs", torch.as_tensor(maxs))   # [K]


        self.base_mu      = nn.Parameter(torch.ones(self.num_keys))
        self.base_log_std = nn.Parameter(torch.ones(self.num_keys) * init_std)
        # )
        self.state_net = nn.GRUCell(2, hidden)
        self.out = nn.Linear(hidden, 2 * self.num_keys)


        self.clamp   = clamp
        self.detach  = detach
        self.max_steps  = float(max_steps)
        self.ema_scale  = float(ema_scale)
        
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.hidden = None 
        
        for name, param in self.state_net.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    def forward(
        self,
        *,
        step:   Tensor | int | None,
        loss:   Tensor | float | None,
        batch_size: int = 1,
        device=None,
    ):
        if device is None:
            device = self.base_mu.device

        # ---------- scalar → Tensor[B] ----------
        step_t = torch.as_tensor(step,  device=device, dtype=torch.float32).view(-1)\
                 if step is not None else torch.zeros(batch_size, device=device)
        loss_t = torch.as_tensor(loss,  device=device, dtype=torch.float32).view(-1)\
                 if loss is not None else torch.zeros(batch_size, device=device)

        if step_t.numel() == 1:   step_t = step_t.expand(batch_size)
        if loss_t.numel() == 1:   loss_t = loss_t.expand(batch_size)


        step_norm = (step_t / self.max_steps).clamp_(0., 1.)          # ∈[0,1]

        loss_norm = torch.tanh(torch.log1p(loss_t) * self.ema_scale)  # ∈(-1,1)


        state_in  = torch.stack([step_norm, loss_norm], dim=-1)       # [B, 2]
        #state_in  = step_norm.unsqueeze(0)
        #delta     = self.state_net(state_in)                          # [B, 2K]
        h_prev = self.hidden
        h_new = self.state_net(state_in, h_prev)          # [B,hidden]
        delta = self.out(h_new)                     # [B,2K]
        
        d_mu, d_ls = delta.split(self.num_keys, dim=-1)

        mu       = self.base_mu      + d_mu
        log_std  = self.base_log_std + d_ls

        if self.clamp:

            log_std = torch.clamp(log_std, -6.0, 2.0)

        std = torch.exp(log_std)


        eps  = torch.randn(batch_size, self.num_keys, device=device)
        raw  = mu + eps * std
        if self.detach:
            raw = raw.detach()


        action = torch.clamp(raw,
                             self.mins.to(device).unsqueeze(0),
                             self.maxs.to(device).unsqueeze(0))

        # ---------- log-prob + entropy ----------
        #logp = (-0.5 * eps.pow(2) - log_std - 0.5 * math.log(2 * math.pi))
        
        logp = (
            -0.5 * (((raw - mu) / std) ** 2)
            - log_std
            - 0.5 * math.log(2 * math.pi)
        )
         
        entropy = log_std + 0.5 * (1. + math.log(2 * math.pi))
        self.hidden = h_new.detach()  
        return {
            k: {
                "action":   action[:, i],    # [B]
                "log_prob": logp  [:, i],
                "entropy":  entropy[:, i],
            }
            for i, k in enumerate(self.keys)
        }
    
    def reset_parameters(self):
        """
        Re‐initialize all weights / biases exactly as in __init__:
          • base_mu → ones
          • base_log_std → init_std
          • every Linear in state_net → He‐init (except last, which is zeroed)
        """
        # 1) Re‐init the base_mu / base_log_std
        with torch.no_grad():
            self.base_mu.data.fill_(1.0)
            self.base_log_std.data.fill_(self.init_std)

        # 2) Walk through state_net and reinitialize all Linear layers
        #    We’ll use the same logic you’d want: He‐uniform for all linears except
        #    the last one, which you want zeroed (weight & bias = 0).
        #    We know `self.state_net[-1]` is the final Linear, so handle it specially.

        # First, re‐zero the final layer exactly as before:
        last_lin: nn.Linear = self.state_net[-1]
        nn.init.zeros_(last_lin.weight)
        nn.init.zeros_(last_lin.bias)

        # Now re‐init every other Linear in state_net
        # (i.e. indices 0 and 2 in your Sequential:  Linear(2,hidden*2) and Linear(hidden*2,hidden) )
        for module in self.state_net[:-1]:
            if isinstance(module, nn.Linear):
                # Kaiming uniform as a reasonable default (fan_in mode, nonlinearity='relu'):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                # For bias, a common choice is zero (or use the default for Kaiming):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)




class ScalarPolicyWithSchedStateGRUKeywiseLog(nn.Module):
    def __init__(
        self,
        key_info_list,
        init_std: float = 0.0,
        hidden: int = 32,
        clamp: bool = True,
        detach: bool = True,
        max_steps: int = 30_000,
        ema_scale: float = 5.0,
        init_mu: float = 0.0,
    ):
        super().__init__()
        self.keys = [name for name, *_ in key_info_list]
        self.num_keys = len(self.keys)
        self.hidden_size = hidden


        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.register_buffer("mins", torch.as_tensor(mins))   # [K]
        self.register_buffer("maxs", torch.as_tensor(maxs))   # [K]


        self.base_mu      = nn.Parameter(torch.zeros(self.num_keys)) 
        self.base_log_std = nn.Parameter(torch.ones(self.num_keys) * init_std)


        self.state_nets = nn.ModuleList([
            nn.GRUCell(2, hidden) for _ in range(self.num_keys)
        ])
        self.out_nets = nn.ModuleList([
            nn.Linear(hidden, 2) for _ in range(self.num_keys)
        ])

        self.clamp = clamp
        self.detach = detach
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


        self.hidden = [None] * self.num_keys

    def set_hidden_state(self):
        self.hidden = [torch.zeros(1, self.hidden_size, device="cuda") for _ in range(self.num_keys)]

    def forward(
        self,
        *,
        step: torch.Tensor | int | None,
        loss: torch.Tensor | float | None,
        batch_size: int = 1,
        device=None,
    ):
        if device is None:
            device = self.base_mu.device

        # scalar → Tensor[B]
        step_t = torch.as_tensor(step, device=device, dtype=torch.float32).view(-1)\
                 if step is not None else torch.zeros(batch_size, device=device)
        loss_t = torch.as_tensor(loss, device=device, dtype=torch.float32).view(-1)\
                 if loss is not None else torch.zeros(batch_size, device=device)
        if step_t.numel() == 1: step_t = step_t.expand(batch_size)
        if loss_t.numel() == 1: loss_t = loss_t.expand(batch_size)


        step_norm = (step_t / self.max_steps).clamp_(0., 1.)
        loss_norm = torch.tanh(torch.log1p(loss_t) * self.ema_scale)
        state_in = torch.stack([step_norm, loss_norm], dim=-1)  # [B,2]


        new_h, d_mu, d_ls = [], [], []
        for i in range(self.num_keys):
            h_prev = self.hidden[i]
            h_i = self.state_nets[i](state_in, h_prev) if h_prev is not None else self.state_nets[i](state_in)
            new_h.append(h_i)
            delta_i = self.out_nets[i](h_i)
            mu_i, ls_i = delta_i.split(1, dim=-1)
            d_mu.append(mu_i)
            d_ls.append(ls_i)


        self.hidden = [h.detach() for h in new_h]

        d_mu = torch.cat(d_mu, dim=1)
        d_ls = torch.cat(d_ls, dim=1)


        mu = self.base_mu + d_mu
        log_std = self.base_log_std + d_ls
        if self.clamp:
            log_std = torch.clamp(log_std, -6.0, 2.0)
        std = torch.exp(log_std)


        eps = torch.randn(batch_size, self.num_keys, device=device)
        log_raw = mu + eps * std  # [B, K]
        raw = torch.exp(log_raw)

        if self.detach:
            raw = raw.detach()


        mins = self.mins.to(device).unsqueeze(0)
        maxs = self.maxs.to(device).unsqueeze(0)
        action = torch.clamp(raw, mins, maxs)


        logp = -0.5 * (((log_raw - mu) / std) ** 2) - log_std - log_raw - 0.5 * math.log(2 * math.pi)
        entropy = log_std + 0.5 * (1. + math.log(2 * math.pi))  # log-space entropy

        return {
            k: {
                "action":   action[:, i],
                "log_prob": logp[:, i],
                "entropy":  entropy[:, i],
            }
            for i, k in enumerate(self.keys)
        }




class ScalarPolicyWithSchedStateGRUKeywiseLog2(nn.Module):
    def __init__(
        self,
        key_info_list,
        init_std: float = 0.0,
        hidden: int = 32,
        clamp: bool = True,
        detach: bool = True,
        max_steps: int = 30_000,
        ema_scale: float = 5.0,
        init_mu: float = 0.0,
    ):
        super().__init__()
        self.keys = [name for name, *_ in key_info_list]
        self.num_keys = len(self.keys)
        self.hidden_size = hidden


        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.register_buffer("mins", torch.as_tensor(mins))   # [K]
        self.register_buffer("maxs", torch.as_tensor(maxs))   # [K]


        self.base_mu      = nn.Parameter(torch.zeros(self.num_keys)) 
        self.base_log_std = nn.Parameter(torch.ones(self.num_keys) * init_std)


        self.state_nets = nn.ModuleList([
            nn.GRUCell(2, hidden) for _ in range(self.num_keys)
        ])
        self.out_nets = nn.ModuleList([
            nn.Linear(hidden, 2) for _ in range(self.num_keys)
        ])

        self.clamp = clamp
        self.detach = detach
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


        self.hidden = [None] * self.num_keys

    def set_hidden_state(self):
        self.hidden = [torch.zeros(1, self.hidden_size, device="cuda") for _ in range(self.num_keys)]

    def forward(
        self,
        *,
        step: torch.Tensor | int | None,
        loss: torch.Tensor | float | None,
        batch_size: int = 1,
        device=None,
    ):
        if device is None:
            device = self.base_mu.device

        # scalar → Tensor[B]
        step_t = torch.as_tensor(step, device=device, dtype=torch.float32).view(-1)\
                 if step is not None else torch.zeros(batch_size, device=device)
        loss_t = torch.as_tensor(loss, device=device, dtype=torch.float32).view(-1)\
                 if loss is not None else torch.zeros(batch_size, device=device)
        if step_t.numel() == 1: step_t = step_t.expand(batch_size)
        if loss_t.numel() == 1: loss_t = loss_t.expand(batch_size)


        step_norm = (step_t / self.max_steps).clamp_(0., 1.)
        loss_norm = torch.tanh(torch.log1p(loss_t) * self.ema_scale)
        state_in = torch.stack([step_norm, loss_norm], dim=-1)  # [B,2]


        new_h, d_mu, d_ls = [], [], []
        for i in range(self.num_keys):
            h_prev = self.hidden[i]
            h_i = self.state_nets[i](state_in, h_prev) if h_prev is not None else self.state_nets[i](state_in)
            new_h.append(h_i)
            delta_i = self.out_nets[i](h_i)
            mu_i, ls_i = delta_i.split(1, dim=-1)
            d_mu.append(mu_i)
            d_ls.append(ls_i)


        self.hidden = [h.detach() for h in new_h]

        d_mu = torch.cat(d_mu, dim=1)
        d_ls = torch.cat(d_ls, dim=1)


        mu = self.base_mu + d_mu
        log_std = self.base_log_std + d_ls
        if self.clamp:
            log_std = torch.clamp(log_std, -6.0, 2.0)
        std = torch.exp(log_std)


        eps = torch.randn(batch_size, self.num_keys, device=device)
        log_raw = mu + eps * std  # [B, K]
        raw = torch.exp(log_raw)

        if self.detach:
            raw = raw.detach()


        #mins = self.mins.to(device).unsqueeze(0)
        #maxs = self.maxs.to(device).unsqueeze(0)
        action = raw #torch.clamp(raw, mins, maxs)


        logp = -0.5 * (((log_raw - mu) / std) ** 2) - log_std - log_raw - 0.5 * math.log(2 * math.pi)
        entropy = log_std + 0.5 * (1. + math.log(2 * math.pi))  # log-space entropy

        return {
            k: {
                "action":   action[:, i],
                "log_prob": logp[:, i],
                "entropy":  entropy[:, i],
            }
            for i, k in enumerate(self.keys)
        }

class ScalarPolicyWithSchedStateGRUKeywise(nn.Module):
    def __init__(
        self,
        key_info_list,
        init_std: float = 0.0,
        hidden: int = 32,
        clamp: bool = True,
        detach: bool = True,
        max_steps: int = 30_000,
        ema_scale: float = 5.0,
        init_mu: float = 1.0
    ):
        super().__init__()
        self.keys     = [name for name, *_ in key_info_list]
        self.num_keys = len(self.keys)
        self.hidden_size = hidden


        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.register_buffer("mins", torch.as_tensor(mins))   # [K]
        self.register_buffer("maxs", torch.as_tensor(maxs))   # [K]


        self.base_mu      = nn.Parameter(init_mu*torch.ones(self.num_keys)) 
        self.base_log_std = nn.Parameter(torch.ones(self.num_keys) * init_std)


        self.state_nets = nn.ModuleList([
            nn.GRUCell(2, hidden) for _ in range(self.num_keys)
        ])
        self.out_nets   = nn.ModuleList([
            nn.Linear(hidden, 2) for _ in range(self.num_keys)
        ])


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
    
    def set_hidden_state(self):
        self.hidden = [torch.zeros(1, self.hidden_size, device="cuda") for k in range(self.num_keys)]
        
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
        step_t = torch.as_tensor(step,  device=device, dtype=torch.float32).view(-1)\
                 if step is not None else torch.zeros(batch_size, device=device)
        loss_t = torch.as_tensor(loss,  device=device, dtype=torch.float32).view(-1)\
                 if loss is not None else torch.zeros(batch_size, device=device)
        if step_t.numel() == 1:   step_t = step_t.expand(batch_size)
        if loss_t.numel() == 1:   loss_t = loss_t.expand(batch_size)


        step_norm = (step_t / self.max_steps).clamp_(0., 1.)
        loss_norm = torch.tanh(torch.log1p(loss_t) * self.ema_scale)
        state_in  = torch.stack([step_norm, loss_norm], dim=-1)  # [B,2]


        new_h   = []
        d_mu    = []
        d_ls    = []
        for i in range(self.num_keys):
            cell   = self.state_nets[i]
            h_prev = self.hidden[i]

            h_i    = cell(state_in, h_prev) if h_prev is not None else cell(state_in)
            new_h.append(h_i)


            delta_i = self.out_nets[i](h_i)
            mu_i, ls_i = delta_i.split(1, dim=-1)
            d_mu.append(mu_i)
            d_ls.append(ls_i)


        self.hidden = [h.detach() for h in new_h]


        d_mu  = torch.cat(d_mu, dim=1)
        d_ls  = torch.cat(d_ls, dim=1)


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

        # log_prob, entropy
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



class ScalarPolicyWithSchedStateGRUKeywiseSquesh(nn.Module):
    """
    每个 key 使用独立的 GRUCell，接口和输出与原类保持一致。and use squesh
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
        init_mu: float = 1.0
    ):
        super().__init__()
        self.keys     = [name for name, *_ in key_info_list]
        self.num_keys = len(self.keys)
        self.hidden_size = hidden


        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.register_buffer("mins", torch.as_tensor(mins))   # [K]
        self.register_buffer("maxs", torch.as_tensor(maxs))   # [K]


        self.base_mu      = nn.Parameter(init_mu*torch.ones(self.num_keys)) 
        self.base_log_std = nn.Parameter(torch.ones(self.num_keys) * init_std)


        self.state_nets = nn.ModuleList([
            nn.GRUCell(2, hidden) for _ in range(self.num_keys)
        ])
        self.out_nets   = nn.ModuleList([
            nn.Linear(hidden, 2) for _ in range(self.num_keys)
        ])


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
    
    def set_hidden_state(self):
        self.hidden = [torch.zeros(1, self.hidden_size, device="cuda") for k in range(self.num_keys)]
        
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
        step_t = torch.as_tensor(step,  device=device, dtype=torch.float32).view(-1)\
                 if step is not None else torch.zeros(batch_size, device=device)
        loss_t = torch.as_tensor(loss,  device=device, dtype=torch.float32).view(-1)\
                 if loss is not None else torch.zeros(batch_size, device=device)
        if step_t.numel() == 1:   step_t = step_t.expand(batch_size)
        if loss_t.numel() == 1:   loss_t = loss_t.expand(batch_size)


        step_norm = (step_t / self.max_steps).clamp_(0., 1.)
        loss_norm = torch.tanh(torch.log1p(loss_t) * self.ema_scale)
        state_in  = torch.stack([step_norm, loss_norm], dim=-1)  # [B,2]


        new_h   = []
        d_mu    = []
        d_ls    = []
        for i in range(self.num_keys):
            cell   = self.state_nets[i]
            h_prev = self.hidden[i]

            h_i    = cell(state_in, h_prev) if h_prev is not None else cell(state_in)
            new_h.append(h_i)


            delta_i = self.out_nets[i](h_i)
            mu_i, ls_i = delta_i.split(1, dim=-1)
            d_mu.append(mu_i)
            d_ls.append(ls_i)


        self.hidden = [h.detach() for h in new_h]


        d_mu  = torch.cat(d_mu, dim=1)
        d_ls  = torch.cat(d_ls, dim=1)


        mu      = self.base_mu + d_mu
        log_std = self.base_log_std + d_ls
        if self.clamp:
            log_std = torch.clamp(log_std, -6.0, 2.0)
        std = torch.exp(log_std)


        eps = torch.randn(batch_size, self.num_keys, device=device)
        raw = mu + eps * std
        if self.detach:
            raw = raw.detach()

        # squash + rescale
        squashed = torch.tanh(raw)
        mins = self.mins.to(device).unsqueeze(0)   # [1, K]
        maxs = self.maxs.to(device).unsqueeze(0)   # [1, K]
        action = mins + 0.5 * (squashed + 1.0) * (maxs - mins)

        # log_prob, entropy
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

class ScalarPolicyWithSchedStateGRUKeywiseNoLoss(nn.Module):
    """
    每个 key 使用独立的 GRUCell，接口和输出与原类保持一致。
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
        init_mu: float = 1.0
    ):
        super().__init__()
        self.keys     = [name for name, *_ in key_info_list]
        self.num_keys = len(self.keys)
        self.hidden_size = hidden


        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.register_buffer("mins", torch.as_tensor(mins))   # [K]
        self.register_buffer("maxs", torch.as_tensor(maxs))   # [K]


        self.base_mu      = nn.Parameter(init_mu*torch.ones(self.num_keys)) 
        self.base_log_std = nn.Parameter(torch.ones(self.num_keys) * init_std)


        self.state_nets = nn.ModuleList([
            nn.GRUCell(2, hidden) for _ in range(self.num_keys)
        ])
        self.out_nets   = nn.ModuleList([
            nn.Linear(hidden, 2) for _ in range(self.num_keys)
        ])


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
    
    def set_hidden_state(self):
        self.hidden = [torch.zeros(1, self.hidden_size, device="cuda") for k in range(self.num_keys)]
        
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
        step_t = torch.as_tensor(step,  device=device, dtype=torch.float32).view(-1)\
                 if step is not None else torch.zeros(batch_size, device=device)
        loss_t = torch.as_tensor(loss,  device=device, dtype=torch.float32).view(-1)\
                 if loss is not None else torch.zeros(batch_size, device=device)
        if step_t.numel() == 1:   step_t = step_t.expand(batch_size)
        if loss_t.numel() == 1:   loss_t = loss_t.expand(batch_size)


        step_norm = (step_t / self.max_steps).clamp_(0., 1.)
        loss_norm = torch.tanh(torch.log1p(loss_t) * self.ema_scale)
        state_in  = torch.stack([step_norm, step_norm], dim=-1)  # [B,2]


        new_h   = []
        d_mu    = []
        d_ls    = []
        for i in range(self.num_keys):
            cell   = self.state_nets[i]
            h_prev = self.hidden[i]

            h_i    = cell(state_in, h_prev) if h_prev is not None else cell(state_in)
            new_h.append(h_i)


            delta_i = self.out_nets[i](h_i)
            mu_i, ls_i = delta_i.split(1, dim=-1)
            d_mu.append(mu_i)
            d_ls.append(ls_i)


        self.hidden = [h.detach() for h in new_h]


        d_mu  = torch.cat(d_mu, dim=1)
        d_ls  = torch.cat(d_ls, dim=1)


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

        # log_prob, entropy
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
import torch
import torch.nn as nn
import math
from typing import List, Union

class MixtureScalarPolicy(nn.Module):
    def __init__(
        self,
        n_mixtures: int,
        key_info_list,
        init_std: float = 0.0,
        hidden: int = 32,
        clamp: bool = True,
        detach: bool = True,
        max_steps: int = 30_000,
        ema_scale: float = 5.0,
    ):
        super().__init__()

        self.policies = nn.ModuleList([
            ScalarPolicyWithSchedStateNN(
                key_info_list,
                init_std=init_std,
                hidden=hidden,
                clamp=clamp,
                detach=detach,
                max_steps=max_steps,
                ema_scale=ema_scale,
            )
            for _ in range(n_mixtures)
        ])

    def forward(
        self,
        index: Union[int, torch.LongTensor],
        step: Union[torch.Tensor, float, None] = None,
        loss: Union[torch.Tensor, float, None] = None,
        batch_size: int = 1,
        device=None,
    ):
        """
        index: 如果是 int，则对整个 batch 调用同一个子网络；
               如果是 LongTensor (shape [B])，则为每个样本指定子网络。
        """
        
        return self.policies[index](
            step=step, loss=loss,
            batch_size=batch_size, device=device
        )





class MixtureGRUPolicy(nn.Module):
    def __init__(
        self,
        n_mixtures: int,
        key_info_list,
        init_std: float = 0.0,
        hidden: int = 32,
        clamp: bool = True,
        detach: bool = True,
        max_steps: int = 30_000,
        ema_scale: float = 5.0,
    ):
        super().__init__()

        self.n_mixtures = n_mixtures
        self.policies = nn.ModuleList([
            ScalarPolicyWithSchedStateGRUKeywise(
                key_info_list,
                init_std=init_std,
                hidden=hidden,
                clamp=clamp,
                detach=detach,
                max_steps=max_steps,
                ema_scale=ema_scale,
            )
            for _ in range(n_mixtures)
        ])
    
    def set_hidden_state(self):
        for i in range(self.n_mixtures):
            self.policies[i].set_hidden_state()
    def forward(
        self,
        index: Union[int, torch.LongTensor],
        step: Union[torch.Tensor, float, None] = None,
        loss: Union[torch.Tensor, float, None] = None,
        batch_size: int = 1,
        device=None,
    ):
        """
        index: 如果是 int，则对整个 batch 调用同一个子网络；
               如果是 LongTensor (shape [B])，则为每个样本指定子网络。
        """
        
        return self.policies[index](
            step=step, loss=loss,
            batch_size=batch_size, device=device
        )

    


# class PerViewScalarPolicyWithLossStatev2(nn.Module):
#     """




#     """
#     def __init__(
#         self,
#         key_info_list,
#         num_views: int,
#         init_std: float = 0.0,
#         clamp: bool = True,
#         detach: bool = True,
#         max_steps: int = 10000,
#         loss_ema_alpha: float = 0.5,
#         ema_scale: float = 5.0,
#         end_max: float = 5.0
#     ):
#         super().__init__()

#         self.keys = [name for name, _, _, _ in key_info_list]
#         self.num_keys = len(self.keys)
#         self.num_views = num_views


#         mins = [min_v / default for _, default, min_v, _ in key_info_list]
#         maxs = [max_v / default for _, default, _, max_v in key_info_list]
#         self.register_buffer('mins',  torch.tensor(mins))  # [K]
#         self.register_buffer('maxs',  torch.tensor(maxs))  # [K]


#         self.mu      = nn.Parameter(torch.ones(num_views, self.num_keys))
#         self.log_std = nn.Parameter(torch.ones(num_views, self.num_keys) * init_std)
#         self.init_std = init_std


#         self.clamp     = clamp
#         self.detach    = detach
#         self.max_steps = max_steps*2


#         self.loss_ema_alpha = loss_ema_alpha
#         self.ema_scale      = ema_scale

#         self.register_buffer('ema_loss',        torch.zeros(1))
#         self.register_buffer('ema_initialized', torch.zeros(1, dtype=torch.bool))


#         self.register_buffer('step_buffer', torch.zeros(1, dtype=torch.long))



#         self.w_mu     = nn.Parameter(torch.zeros(self.num_keys))
#         self.w_logstd = nn.Parameter(torch.zeros(self.num_keys))
#         self.end_max = end_max

#     def forward(
#         self,
#         view_idx,
#         batch_size: int = 1,
#         device=None,


#     ):
#         if device is None:
#             device = self.mu.device


#         if step is not None:
#             self.step_buffer.copy_(torch.tensor(step, dtype=torch.long, device=device))


#         if loss is not None:

#             loss_val = loss.detach().mean()
#             if not self.ema_initialized:

#                 self.ema_loss[:]        = loss_val
#                 self.ema_initialized[:] = torch.tensor(True, device=device)
#             else:

#                 self.ema_loss[:] = (
#                     self.loss_ema_alpha * self.ema_loss +
#                     (1 - self.loss_ema_alpha) * loss_val
#                 )
#             # centered & scale → sigmoid
#             centered = loss_val - self.ema_loss
#             loss_norm = torch.sigmoid(centered * self.ema_scale)
#         else:
#             loss_norm = torch.tensor(0.0, device=device)


#         mu_v     = self.mu[view_idx].to(device)      # [K] or [B,K]
#         logstd_v = self.log_std[view_idx].to(device) # [K] or [B,K]


#         if self.clamp:
#             min_ls, init_max, end_max = -20.0, 1.0, self.end_max # default -5.0
#             #min_ls, init_max, end_max = -20.0, 1.0, 1.0 
#             #min_ls, init_max, end_max = -10.0, 2.0, 1.0  #


#             prog = float(self.step_buffer.item()) / float(self.max_steps)
#             prog = min(max(prog, 0.0), 1.0)
#             dyn_max = init_max + prog * (end_max - init_max)
#             logstd_v = torch.clamp(logstd_v, min_ls, dyn_max)


#         w_mu  = self.w_mu.to(device)
#         w_ls  = self.w_logstd.to(device)
#         mu_v     = mu_v     + loss_norm * w_mu
#         logstd_v = logstd_v + loss_norm * w_ls


#         std_v = torch.exp(logstd_v)

#         if mu_v.dim() == 1:
#             mu_v     = mu_v.unsqueeze(0).expand(batch_size, -1)
#             std_v    = std_v.unsqueeze(0).expand(batch_size, -1)
#             logstd_v = logstd_v.unsqueeze(0).expand(batch_size, -1)

#         eps = torch.randn_like(mu_v)
#         raw = mu_v + eps * std_v
#         if self.detach:
#             raw = raw.detach()
            
        
        
#         logp = (
#             -0.5 * (((raw - mu_v) / std_v) ** 2)
#             - logstd_v
#             - 0.5 * math.log(2 * math.pi)
#         )


#         # H = ½(1 + ln(2π)) + ln σ
#         const = 0.5 * (1.0 + math.log(2.0 * math.pi))
#         entropy_v = logstd_v + const  # [B, K]


#         min_v = self.mins.to(device).unsqueeze(0).expand_as(raw)
#         max_v = self.maxs.to(device).unsqueeze(0).expand_as(raw)
#         action = torch.clamp(raw, min_v, max_v)


#         return {
#             k: {
#                 "action":   action[:, i],   # [B]
#                 "log_prob": logp[:, i],     # [B]
#                 "entropy":  entropy_v[:, i] # [B]
#             }
#             for i, k in enumerate(self.keys)
#         }
        
        
        


from torch.distributions import Normal

class PerViewScalarPolicyWithLossStatev2(nn.Module):
    """
    在线调度版本，保持原接口不变：
      forward(view_idx, batch_size=1, device=None, loss=None, step=None)
    返回同样的 {key: {"action", "log_prob", "entropy"}} 结构，
    但 μ/logσ 不再是纯参数，而是由 (step_norm, loss_norm) + view_idx embedding
    经由一个 MLP 动态生成。
    """
    def __init__(
        self,
        key_info_list,
        num_views: int,
        init_std: float = 0.0,
        clamp: bool = True,
        detach: bool = True,
        max_steps: int = 10000,
        loss_ema_alpha: float = 0.5,
        ema_scale: float = 5.0,
        end_max: float = 5.0,

        embed_dim: int = 16,
        hidden_sizes=(64, 64),
    ):
        super().__init__()

        self.keys = [name for name, _, _, _ in key_info_list]
        self.num_keys = len(self.keys)
        self.num_views = num_views


        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.register_buffer('mins', torch.tensor(mins))  # [K]
        self.register_buffer('maxs', torch.tensor(maxs))  # [K]


        self.mu_base   = nn.Parameter(torch.ones(num_views, self.num_keys))
        self.logstd_base = nn.Parameter(torch.ones(num_views, self.num_keys) * init_std)


        self.clamp     = clamp
        self.detach    = detach
        self.max_steps = max_steps * 2
        self.end_max   = end_max


        self.loss_ema_alpha = loss_ema_alpha
        self.ema_scale      = ema_scale
        self.register_buffer('ema_loss',        torch.zeros(1))
        self.register_buffer('ema_initialized', torch.zeros(1, dtype=torch.bool))


        self.register_buffer('step_buffer', torch.zeros(1, dtype=torch.long))



        state_dim = 2
        # view embedding
        self.view_embed = nn.Embedding(num_views, embed_dim)

        input_dim = embed_dim + state_dim
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 2 * self.num_keys))
        self.state_net = nn.Sequential(*layers)

        final = self.state_net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        view_idx,
        batch_size: int = 1,
        device=None,
        loss: torch.Tensor = None,
        step: int = None,
    ):
        if device is None:
            device = self.mu_base.device


        if step is not None:
            self.step_buffer.fill_(step)


        if loss is not None:
            lv = loss.detach().mean()
            if not self.ema_initialized:
                self.ema_loss[:]        = lv
                self.ema_initialized[:] = torch.tensor(True, device=device)
            else:
                self.ema_loss[:] = (
                    self.loss_ema_alpha * self.ema_loss +
                    (1 - self.loss_ema_alpha) * lv
                )
            centered = lv - self.ema_loss
            loss_norm = torch.sigmoid(centered * self.ema_scale)
        else:
            loss_norm = torch.tensor(0.0, device=device)


        # step_norm ∈ [0,1]
        prog = float(self.step_buffer.item()) / float(self.max_steps)
        prog = min(max(prog, 0.0), 1.0)
        step_norm = torch.tensor(prog, device=device)


        if isinstance(view_idx, int):
            v_idx = torch.full((batch_size,), view_idx, dtype=torch.long, device=device)
        else:
            v_idx = view_idx.to(device)

        # state_feat: [B,2]
        state_feat = torch.stack([
            step_norm.expand(batch_size),
            loss_norm.expand(batch_size)
        ], dim=-1)  # [B,2]

        # view embedding: [B,embed_dim]
        v_emb = self.view_embed(v_idx)  # [B, embed_dim]


        h = torch.cat([v_emb, state_feat], dim=-1)  # [B, embed_dim+2]
        out = self.state_net(h)                     # [B, 2*K]
        delta_mu, delta_logstd = out.chunk(2, dim=-1)


        

        mu_b  = self.mu_base[v_idx]       # [B,K]
        ls_b  = self.logstd_base[v_idx]   # [B,K]
        mu_v     = mu_b  + delta_mu
        logstd_v = ls_b + delta_logstd


        if self.clamp:
            min_ls, init_max, end_max = -20.0, 1.0, self.end_max
            prog = float(self.step_buffer.item()) / float(self.max_steps)
            prog = min(max(prog, 0.0), 1.0)
            dyn_max = init_max + prog * (end_max - init_max)
            logstd_v = torch.clamp(logstd_v, min_ls, dyn_max)

        std_v = torch.exp(logstd_v)  # [B,K]


        eps = torch.randn_like(mu_v)
        raw = mu_v + eps * std_v
        if self.detach:
            raw = raw.detach()

        # log_prob & entropy
        dist = Normal(mu_v, std_v)

        logp = dist.log_prob(raw).sum(-1, keepdim=True)  # [B,1]
        ent  = dist.entropy().sum(-1, keepdim=True)      # [B,1]


        min_v = self.mins.to(device).unsqueeze(0).expand_as(raw)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(raw)
        action = torch.clamp(raw, min_v, max_v)  # [B,K]


        out_dict = {}
        for i, k in enumerate(self.keys):
            out_dict[k] = {
                "action":   action[:, i],  # [B]
                "log_prob": logp[:, 0],    # [B]
                "entropy":  ent[:, 0],     # [B]
            }
        return out_dict
    
    



class CoordinateWiseRLPolicy(nn.Module):
    """
    Coordinate‑wise Gaussian policy，按视图（view）输出每个 key 的 lr 分布。
    返回格式跟你原来的一模一样：
      {
        key1: {"action": Tensor[B], "log_prob": Tensor[B], "entropy": Tensor[B]},
        key2: {...}, …
      }
    """
    def __init__(self,
                 key_info_list,
                 num_views: int,
                 hidden_size: int = 20,
                 init_log_std: float = -3.0):
        super().__init__()

        self.keys      = [name for name,_,_,_ in key_info_list]
        self.num_keys  = len(self.keys)
        self.num_views = num_views
        self.hidden   = hidden_size

        # —— shared LSTMCell core —— 
        self.lstm = nn.LSTMCell(input_size=1, hidden_size=hidden_size)


        self.mu_head      = nn.Linear(hidden_size, 1)
        self.logstd_head = nn.Linear(hidden_size, 1)

        nn.init.constant_(self.logstd_head.bias, init_log_std)


        # shape = [num_views, num_keys, hidden_size]
        self.register_buffer(
            'h', torch.zeros(num_views, self.num_keys, hidden_size)
        )
        self.register_buffer(
            'c', torch.zeros(num_views, self.num_keys, hidden_size)
        )

    def forward(self, view_idx: int, grads: torch.Tensor):
        """
        Args:
          view_idx: 整数，当前视图索引
          grads:    Tensor([B, K]) 或 ([K])  —— 该视图下每个 key 的梯度
        Returns: dict keyed by self.keys，内部同你原来的格式
        """

        if grads.dim() == 1:
            # [K] → [1, K]
            grads = grads.unsqueeze(0)
        B, K = grads.shape
        assert K == self.num_keys, "grads shape must match num_keys"


        # h0,c0 shape = [K, hidden]
        h0 = self.h[view_idx]
        c0 = self.c[view_idx]

        h_in = h0.unsqueeze(0).expand(B, -1, -1).contiguous().view(B*K, -1)
        c_in = c0.unsqueeze(0).expand(B, -1, -1).contiguous().view(B*K, -1)


        x = grads.contiguous().view(B*K, 1)


        h_next, c_next = self.lstm(x, (h_in, c_in))


        new_h = h_next.view(B, K, self.hidden).mean(dim=0)  # [K, hidden]
        new_c = c_next.view(B, K, self.hidden).mean(dim=0)
        self.h[view_idx].copy_(new_h)
        self.c[view_idx].copy_(new_c)


        mu_raw      = self.mu_head(h_next)      # [B*K, 1]
        logstd_raw = self.logstd_head(h_next)  # [B*K, 1]
        mu      = mu_raw.view(B, K)            # [B, K]
        logstd = logstd_raw.view(B, K)
        std     = torch.exp(logstd)


        eps   = torch.randn_like(std)
        alpha = mu + eps * std                 # [B, K]


        # logp = -0.5 * ((α-μ)/σ)^2 - logσ - 0.5*ln(2π)
        logp = -0.5 * ((alpha - mu)/std).pow(2) - logstd - 0.5 * math.log(2*math.pi)
        # entropy = ½(1 + ln(2π)) + lnσ
        entropy = 0.5*(1.0 + math.log(2.0*math.pi)) + logstd  # [B, K]


        out = {}
        for i, k in enumerate(self.keys):
            out[k] = {
                "action":   alpha[:, i],   # [B]
                "log_prob": logp[:, i],     # [B]
                "entropy":  entropy[:, i]   # [B]
            }
        return out
    
    
    


import torch
import torch.nn as nn
import math

class ScalarRLPolicyWithKeys(nn.Module):
    """
    一个只输出标量学习率的 RL 策略，
    但接口保持和你原来完全一样：
      return {
        key1: {"action": Tensor[B], "log_prob": Tensor[B], "entropy": Tensor[B]},
        key2: {...}, …
      }
    """
    def __init__(self,
                 key_info_list,
                 num_views: int,
                 hidden_size: int = 32,
                 init_log_std: float = -3.0,
                 max_steps: int = 10000):
        super().__init__()

        self.keys      = [name for name,_,_,_ in key_info_list]
        self.num_keys  = len(self.keys)
        self.num_views = num_views
        self.max_steps = max_steps


        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

        with torch.no_grad():
            self.net[-1].bias[1].fill_(init_log_std)

    def forward(self,
                view_idx: int,
                loss: torch.Tensor,
                step: int):
        """
        Args:
          view_idx: 当前视图索引（保留这个参数仅为兼容）
          loss:     当前 loss 标量或 [B]
          step:     当前训练步数
        Returns: dict, 完全和你原来的一样
        """

        if loss.dim() == 0:
            loss = loss.unsqueeze(0)          # [1]
        B = loss.shape[0]


        loss_norm = loss.detach().view(B, 1)            # [B,1]
        step_norm = torch.full((B,1), step / self.max_steps,
                               device=loss.device)      # [B,1]

        x = torch.cat([loss_norm, step_norm], dim=1)     # [B,2]


        mu_logstd = self.net(x)         # [B,2]
        mu      = mu_logstd[:, 0]       # [B]
        logstd = mu_logstd[:, 1]
        std     = torch.exp(logstd)


        eps = torch.randn_like(std)
        lr  = mu + eps * std            # [B]


        logp = -0.5 * ((lr - mu)/std).pow(2)\
               - logstd\
               - 0.5 * math.log(2*math.pi)      # [B]
        ent  = 0.5*(1.0 + math.log(2.0*math.pi)) + logstd  # [B]


        out = {}
        for k in self.keys:
            out[k] = {
                "action":   lr,      # [B]
                "log_prob": logp,    # [B]
                "entropy":  ent      # [B]
            }
        return out





class DeepPerViewScalarPolicyWithLossState(nn.Module):
    """
    在 PerViewScalarPolicyWithLossState 基础上：
      - 增加一个小型 MLP：输入 (view embedding, loss_norm, step_norm)
        → 输出每个 key 的 μ 偏移量和 logσ 偏移量
      - 基础 μ/logσ 仍来自 self.mu, self.log_std，可视为 residual base
      - 其余采样、clamp、detach、返回接口不变
    """
    def __init__(
        self,
        key_info_list,
        num_views: int,
        init_std: float = 0.0,
        clamp: bool = True,
        detach: bool = True,
        max_steps: int = 10000,
        loss_ema_alpha: float = 0.5,
        ema_scale: float = 5.0,

        embed_dim: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.keys      = [name for name, _, _, _ in key_info_list]
        self.num_keys  = len(self.keys)
        self.num_views = num_views

        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.register_buffer('mins',  torch.tensor(mins))  # [K]
        self.register_buffer('maxs',  torch.tensor(maxs))  # [K]


        self.mu      = nn.Parameter(torch.ones(num_views, self.num_keys))
        self.log_std = nn.Parameter(torch.ones(num_views, self.num_keys) * init_std)

        self.clamp     = clamp
        self.detach    = detach
        self.max_steps = max_steps * 2


        self.loss_ema_alpha = loss_ema_alpha
        self.ema_scale      = ema_scale
        self.register_buffer('ema_loss',        torch.zeros(1))
        self.register_buffer('ema_initialized', torch.zeros(1, dtype=torch.bool))
        self.register_buffer('step_buffer',     torch.zeros(1, dtype=torch.long))


        self.view_emb = nn.Embedding(num_views, embed_dim)
        mlp_input = embed_dim + 2  # [view_emb, loss_norm, step_norm]
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input,   hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,  hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * self.num_keys)
        )

    def forward(
        self,
        view_idx: int,
        batch_size: int = 1,
        device=None,
        loss: torch.Tensor = None,
        step: int = None,
    ):
        if device is None:
            device = self.mu.device


        if step is not None:
            self.step_buffer.copy_(torch.tensor(step, device=device))
        if loss is not None:
            loss_val = loss.detach().mean()
            if not self.ema_initialized:
                self.ema_loss[:]        = loss_val
                self.ema_initialized[:] = torch.tensor(True, device=device)
            else:
                self.ema_loss[:] = (
                    self.loss_ema_alpha * self.ema_loss +
                    (1 - self.loss_ema_alpha) * loss_val
                )
            centered  = loss_val - self.ema_loss
            loss_norm = torch.sigmoid(centered * self.ema_scale)
        else:
            loss_norm = torch.tensor(0.0, device=device)



        if loss_norm.dim() == 0:
            loss_norm = loss_norm.unsqueeze(0)
        B = loss_norm.size(0)
        step_norm = torch.full((B,1),
            step / self.max_steps, device=device)

        # view embedding [view_idx] → [E], expand to [B, E]
        v_emb = self.view_emb(
            torch.tensor(view_idx, device=device)
        ).unsqueeze(0).expand(B, -1)  # [B, embed_dim]

        mlp_in = torch.cat([v_emb, loss_norm.unsqueeze(-1), step_norm], dim=1)  # [B, E+2]
        offset = self.mlp(mlp_in)  # [B, 2*K]
        mu_off, logstd_off = offset.split(self.num_keys, dim=1)



        base_mu     = self.mu[view_idx].unsqueeze(0).expand(B, -1)
        base_logstd = self.log_std[view_idx].unsqueeze(0).expand(B, -1)

        mu_v     = base_mu     + mu_off     # [B, K]
        logstd_v = base_logstd + logstd_off # [B, K]
        std_v    = torch.exp(logstd_v)


        eps = torch.randn_like(mu_v)
        raw = mu_v + eps * std_v
        if self.detach:
            raw = raw.detach()

        min_v = self.mins.to(device).unsqueeze(0).expand_as(raw)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(raw)
        action = torch.clamp(raw, min_v, max_v)

        # —— 5) log_prob + entropy —— 
        logp = (
            -0.5 * ((action - mu_v)/std_v).pow(2)
            - logstd_v
            - 0.5 * math.log(2 * math.pi)
        )
        const = 0.5 * (1.0 + math.log(2.0 * math.pi))
        entropy_v = logstd_v + const  # [B, K]


        return {
            k: {
                "action":   action[:, i],   # [B]
                "log_prob": logp[:, i],     # [B]
                "entropy":  entropy_v[:, i] # [B]
            }
            for i, k in enumerate(self.keys)
        }
        
        
    



class DeepPerViewScalarPolicyWithLossStateV2(nn.Module):
    """
    在 PerViewScalarPolicyWithLossState 基础上：
      - 增加一个小型 MLP：输入 (view embedding, loss_norm, step_norm)
        → 直接输出每个 key 的 μ 和 logσ
      - 其余采样、clamp、detach、输出接口不变
    """
    def __init__(
        self,
        key_info_list,
        num_views: int,
        init_std: float = 0.0,
        clamp: bool = True,
        detach: bool = True,
        max_steps: int = 10000,
        loss_ema_alpha: float = 0.5,
        ema_scale: float = 5.0,

        embed_dim: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.keys      = [name for name, _, _, _ in key_info_list]
        self.num_keys  = len(self.keys)
        self.num_views = num_views

        mins = [min_v / default for _, default, min_v, _ in key_info_list]
        maxs = [max_v / default for _, default, _, max_v in key_info_list]
        self.register_buffer('mins',  torch.tensor(mins))   # [K]
        self.register_buffer('maxs',  torch.tensor(maxs))   # [K]

        self.clamp     = clamp
        self.detach    = detach
        self.max_steps = max_steps * 2
        self.loss_clip = 5.0


        self.loss_ema_alpha = loss_ema_alpha
        self.ema_scale      = ema_scale
        self.register_buffer('ema_loss',        torch.zeros(1))
        self.register_buffer('ema_initialized', torch.zeros(1, dtype=torch.bool))
        self.register_buffer('step_buffer',     torch.zeros(1, dtype=torch.long))


        self.view_emb = nn.Embedding(num_views, embed_dim)
        mlp_input = embed_dim + 2  # [view_emb, loss_norm, step_norm]
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input,   hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,  hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * self.num_keys)
        )

    def forward(
        self,
        view_idx: int,
        batch_size: int = 1,
        device=None,
        loss: torch.Tensor = None,
        step: int = None,
    ):
        if device is None:
            device = next(self.parameters()).device


        if step is not None:
            self.step_buffer.copy_(torch.tensor(step, device=device))
        if loss is not None:
            loss_val = loss.detach().mean()
            loss_val = loss_val.clamp(min=0.0, max=0.15)

            if not self.ema_initialized:
                self.ema_loss[:]        = loss_val
                self.ema_initialized[:] = torch.tensor(True, device=device)
            else:
                self.ema_loss[:] = (
                    self.loss_ema_alpha * self.ema_loss +
                    (1 - self.loss_ema_alpha) * loss_val
                )
            centered  = loss_val - self.ema_loss
            loss_norm = torch.sigmoid(centered * self.ema_scale)

        else:
            loss_norm = torch.tensor(0.0, device=device)


        if loss_norm.dim() == 0:
            loss_norm = loss_norm.unsqueeze(0)
        B = loss_norm.size(0)
        step_norm = torch.full((B,1), step / self.max_steps, device=device)

        v_emb = self.view_emb(
            torch.tensor(view_idx, device=device)
        ).unsqueeze(0).expand(B, -1)  # [B, embed_dim]

        mlp_in = torch.cat([v_emb, loss_norm.unsqueeze(-1), step_norm], dim=1)  # [B, E+2]
        out = self.mlp(mlp_in)             # [B, 2*K]
        mu_v, logstd_v = out.split(self.num_keys, dim=1)
        std_v = torch.exp(logstd_v)


        eps   = torch.randn_like(mu_v)
        raw   = mu_v + eps * std_v
        if self.detach:
            raw = raw.detach()

        min_v = self.mins.to(device).unsqueeze(0).expand_as(raw)
        max_v = self.maxs.to(device).unsqueeze(0).expand_as(raw)
        action = torch.clamp(raw, min_v, max_v)

        # —— 4) log_prob + entropy —— 
        logp = (
            -0.5 * ((action - mu_v)/std_v).pow(2)
            - logstd_v
            - 0.5 * math.log(2 * math.pi)
        )
        const = 0.5 * (1.0 + math.log(2.0 * math.pi))
        entropy_v = logstd_v + const  # [B, K]


        return {
            k: {
                "action":   action[:, i],   # [B]
                "log_prob": logp[:, i],     # [B]
                "entropy":  entropy_v[:, i] # [B]
            }
            for i, k in enumerate(self.keys)
        }




class PolicyNetwork(nn.Module):
    """
    将状态向量 s_t 映射到高斯动作分布 dist(a|s).

    Args:
        state_dim (int): 输入状态维度（s_t 的长度）
        action_dim (int): 输出动作维度（你要更新的参数数量或分组数量）
        hidden_sizes (list of int): 隐藏层神经元个数列表
        log_std_min (float): 最小 log 标准差
        log_std_max (float): 最大 log 标准差
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: list = [256, 256],
                 log_std_min: float = -20,
                 log_std_max: float = 2):
        super().__init__()

        dims = [state_dim] + hidden_sizes
        layers = []
        for i in range(len(hidden_sizes)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)


        self.mean_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state: torch.Tensor):
        """
        state: [batch?, state_dim]
        返回一个 Multivariate Normal 分布
        """
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std,
                              self.log_std_min,
                              self.log_std_max)
        std = torch.exp(log_std)


        base_dist = Normal(mean, std)
        dist = Independent(base_dist, 1)
        return dist