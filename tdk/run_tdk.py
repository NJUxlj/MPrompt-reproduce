import os, sys
import os.path as osp
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
# os.chdir() 是 Python 内置函数，用于改变当前的工作目录。此处将当前工作目录更改为脚本所在的目录。sys.path[0] 表示脚本的绝对路径，此操作确保程序从脚本所在目录开始运行。
os.chdir(sys.path[0])
# sys.path.append("..")：将当前目录的上一级目录添加到 Python 的模块搜索路径中。这样程序就可以导入上一级目录中的模块。
sys.path.append("..")