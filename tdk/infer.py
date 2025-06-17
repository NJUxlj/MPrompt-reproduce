import os, sys
os.chdir(sys.path[0])  # ����ǰ����Ŀ¼�л����ű����ڵ�Ŀ¼
sys.path.append("..")  # ����һ��Ŀ¼��ӵ�Python��ģ������·����,�Ա���Ե����ϼ�Ŀ¼�е�ģ��
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