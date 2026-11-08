import torch
from torch.optim import Adam

class AdamUndo(Adam):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self._backup_params = None

    def step(self, closure=None):
        # Before updating, back up parameters and their gradients.
        backup = []
        for group in self.param_groups:
            group_backup = []
            for param in group['params']:
                if param.requires_grad:
                    # Save a copy of the current parameter data and its gradient (if exists)
                    param_backup = {
                        'param': param,
                        'data': param.data.clone(),
                        'grad': param.grad.data.clone() if param.grad is not None else None,
                    }
                    group_backup.append(param_backup)
            backup.append(group_backup)
        self._backup_params = backup

        # Perform the Adam update (using the built-in functionality)
        loss = super().step(closure)
        return loss

    def undo(self):
        # Check if backup exists.
        if self._backup_params is None:
            raise RuntimeError("No backup found. Call step() before calling undo_last_update().")
        
        # Restore parameter data and gradients from backup.
        for group_backup in self._backup_params:
            for backup in group_backup:
                param = backup['param']
                param.data.copy_(backup['data'])
                # Restore gradient if it was available
                if backup['grad'] is not None:
                    if param.grad is None:
                        param.grad = backup['grad'].clone()
                    else:
                        param.grad.data.copy_(backup['grad'])
        
        # Clear the backup after undoing.
        self._backup_params = None

# Example usage:
if __name__ == '__main__':
    # A simple model and data
    model = torch.nn.Linear(10, 1)
    optimizer = AdamUndo(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Dummy input and target
    x = torch.randn(5, 10)
    target = torch.randn(5, 1)

    # Forward pass, loss computation, and backward pass
    output = model(x)
    loss = criterion(output, target)
    loss.backward()

    # Perform an optimizer step (and store backup internally)
    optimizer.step()

    # At this point, you can decide to undo the update.
    optimizer.undo()
    
    # Now, model parameters and gradients have been restored to their values before the last update.
    print("Undo complete.")
