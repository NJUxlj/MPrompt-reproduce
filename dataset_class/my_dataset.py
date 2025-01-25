import logging
import torch 
import pandas as pd
import os.path as osp
import os, json, string
import numpy as np
import re


logger = logging.getLogger(__name__)



class tdk_QA_dataset:
    def __init__(self, args, tokenizer, data_split = 'train'):
        self.args = args
        self.file_path = os.path.join()
        self.tokenizer = tokenizer
        self.ques_len = args.max_ques_length
        self.cont_len = args.max_cont_length
        self.ans_len = args.max_ans_length
        self.data_split = data_split

        self.read_file(args)

    def read_file(self, args):
        pass


    def __len__(self):
        pass



    def __getitem__(self, index):
        pass


