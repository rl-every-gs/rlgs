"""Read-only checkpoint loader for the frozen RobustNeRF EMA-GS implementation.

The training class saves (GaussianModel.capture(), filter_3D, step), but its
standalone _setup path never restores that tuple. This inference-only adapter
restores the saved tensors and delegates rendering to the original method.
It does not initialize training optimizers, masks, DINO weights, or RL policies.
"""
from pathlib import Path
import shutil

import torch
import numpy as np

from asym_emags import AsymmetricGS
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state


class CheckpointAsymmetricGS(AsymmetricGS):
    def save(self, path):
        # NerfBaselines calls save() to hash the inference model. Reproduce the
        # original snapshot bytes; no optimizer or EMA state is synthesized.
        destination = Path(path).resolve()
        source = Path(self.checkpoint).resolve()
        if destination == source or destination.is_relative_to(source):
            raise RuntimeError("Inference snapshots must be saved outside the original checkpoint.")
        shutil.copytree(source, destination, dirs_exist_ok=True)

    def _setup(self, train_dataset):
        if train_dataset is not None or self.checkpoint is None:
            raise RuntimeError("This adapter only evaluates existing checkpoints; use asym_emags for training.")
        root = Path(self.checkpoint)
        if list(root.glob('appearance_transform-*.pth')) or list(root.glob('global_appearance_1-*.pth')):
            raise RuntimeError("Appearance-conditioned PhotoTourism checkpoints require a separate evaluation adapter. This entry point supports the frozen RobustNeRF checkpoints.")
        safe_state(False)
        filename = root / f'chkpnt-{self._loaded_step}.pth'
        # capture() includes a NumPy spatial_lr_scale scalar. Permit just the
        # numeric scalar/dtype globals needed by those old tensors, retaining
        # PyTorch's weights-only loader instead of unrestricted pickle loading.
        with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype('float64')), type(np.dtype('float32'))]):
            saved = torch.load(filename, map_location='cuda', weights_only=True)
        if not isinstance(saved, (tuple, list)) or len(saved) != 3:
            raise RuntimeError(f"Unsupported checkpoint format: {filename}")
        captured, filter_3d, step = saved
        if not isinstance(captured, (tuple, list)) or len(captured) != 13:
            raise RuntimeError("Expected the frozen GaussianModel.capture() tuple with 13 fields.")
        if int(step) != self._loaded_step:
            raise RuntimeError("Checkpoint filename and saved step disagree.")
        model = GaussianModel(self.dataset.sh_degree, enable_ema=False)
        names = ('active_sh_degree', '_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity', 'max_radii2D', 'xyz_gradient_accum', 'denom', None, 'spatial_lr_scale', '_local_encoding')
        for name, value in zip(names, captured):
            if name is not None:
                setattr(model, name, value)
        if not isinstance(filter_3d, torch.Tensor) or filter_3d.shape != (model.get_xyz.shape[0], 1):
            raise RuntimeError("Saved 3D filter does not match the Gaussian count.")
        if not torch.isfinite(filter_3d).all():
            raise RuntimeError("Saved 3D filter contains non-finite values.")
        model.filter_3D = filter_3d
        self.gaussians_1 = model
        self.step = int(step)
        self.use_color_transform = False
        self.background = torch.tensor([1., 1., 1.] if self.dataset.white_background else [0., 0., 0.], device='cuda')
        self.trainCameras_1 = None
        self.highresolution_index_1 = None
        self.rlgs = None

