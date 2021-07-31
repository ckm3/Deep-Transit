FRAMEWORK="pytorch" # megengine
if FRAMEWORK=="pytorch":
    """
    str: Device of your computer. It should be "cuda" for GPU or "cpu" or CPU.
    Default we will automatically choose GPU or CPU based on whether you CUDA is available.
    """
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 416
"""
int: Image size of you training data set, should be the multiples of 32
"""
LEARNING_RATE = 1e-4
"""
float: Learning rate parameter of the Adam optimization
"""
WEIGHT_DECAY = 1e-4
"""
float: Weight decay parameter of the Adam optimization
"""
NUM_EPOCHS = 100
"""
int: Max number of epochs of you training
"""
BATCH_SIZE = 96
"""
int: Batch size of you training
"""
CONF_THRESHOLD = 0.6
"""
float: The confidence threshold for validation and detection.
The number should be 0 to 1, it can be a represent of S/N
"""
MAP_IOU_THRESH = 0.5
"""
float: The IOU threshold of the AP calculation
"""
NMS_IOU_THRESH = 0.1
"""
float: The IOU threshold of the NMS calculation
"""
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
"""
list: The feature size of three scales
"""
NUM_WORKERS = 0
"""
int: Number of works for loading data set
"""
PIN_MEMORY = True
"""
bool: 
"""
LOAD_MODEL = False
"""
bool: If true, it will require loading a previous checkpoint before training
"""
SAVE_MODEL = False
"""
bool: If true, it will save model during the training step
"""
ENABLE_AMP = True
"""
bool: If true, AMP (automatic mixed precision) will be enabled.
Note: if AMD CPU, the ENABLE_AMP should be False.
"""
ENABLE_WANDB = False
"""
bool: If true, a Weights & Biases offline stroage will be enabled
"""
CHECKPOINT_FILE = "checkpoint.pth.tar"
"""
str: The path of checkpoint file
"""
DATASET = 'Data'
"""
str: The path of data set
"""
IMG_DIR = DATASET + "/transit-images/"
"""
str: The path of image directory
"""
LABEL_DIR = DATASET + "/transit-labels/"
"""
str: The path of label directory
"""

ANCHORS = [
    [(0.1, 0.9)],
    [(0.05, 0.7)],
    [(0.02, 0.3)],
]

def data_transforms():
    if FRAMEWORK == "pytorch":
        from torchvision import transforms
        return transforms.Compose([
    transforms.ToTensor(),
])
    else:
        def bypass(x):return x
        return bypass

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
