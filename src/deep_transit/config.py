import torch
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from .utils import seed_everything
seed_everything()  # If you want deterministic behavior
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
IMAGE_SIZE = 416
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 48
CONF_THRESHOLD = 0.5
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.1
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
ENABLE_AMP = True
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = False
ENABLE_WANDB = False
CHECKPOINT_FILE = "checkpoint.pth.tar"

DATASET = 'Data'
IMG_DIR = DATASET + "/transit-images/"
LABEL_DIR = DATASET + "/transit-labels/"

ANCHORS = [
    [(0.1, 0.9)],
    [(0.05, 0.7)],
    [(0.02, 0.3)],
]

data_transforms = transforms.Compose([
    transforms.ToTensor(),
])

model_config = [
    (32, 3, 1),
    (64, 3, 2),
    ["R", 1],
    (128, 3, 2),
    ["R", 2],
    (256, 3, 2),
    ["R", 8],
    (512, 3, 2),
    ["R", 8],
    (1024, 3, 2),
    ["R", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]
