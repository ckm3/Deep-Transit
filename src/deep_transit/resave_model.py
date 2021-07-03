from . import config
from .utils import load_checkpoint
from .model import YOLOv3
import torch.optim as optim
import torch


def save_checkpoint_to_model(checkpoint_path, model_path):
    """
    Save trained checkpoint to models
    Parameters
    ----------
    checkpoint_path : str
    model_path : str
    """
    model = YOLOv3().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True,
                                                        cooldown=3)
    _ = load_checkpoint(checkpoint_path, model, optimizer, config.LEARNING_RATE, lr_scheduler)
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    save_checkpoint_to_model()
