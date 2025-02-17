import os
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# change this depending on where you prefer to store datasets and logs
DATA_PATH = ROOT_DIR    
LOGDIR_PATH_PREFIX = ROOT_DIR