import os, sys
os.chdir(sys.path[0])  # 将当前工作目录切换到脚本所在的目录
sys.path.append("..")  # 将上一级目录添加到Python的模块搜索路径中,以便可以导入上级目录中的模块
import os.path as osp
import argparse
import time
import json
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.log_utils import log_params
from evaluation.evalution import evaluate
from model_class.tdk_model import T5tdkForConditionalGeneration
from utils.tdk_utils import load_tdkdata

logger = logging.getLogger(__name__)