"""Register the inference-only loader under the checkpoint's saved method id."""
from nerfbaselines import register

register({
    'id': 'asym_emags',
    'method_class': 'evaluation_model:CheckpointAsymmetricGS',
})
