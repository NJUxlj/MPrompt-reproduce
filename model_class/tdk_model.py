# 原始版本，只在encoder加，有self attention和MLP两种模式

import sys
import torch
from torch import nn
import logging

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Config
)


from .knowledge_prompt import knowledge_prompt

logger = logging.getLogger(__name__)




class T5tdkForConditionalGeneration(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path = args.model_name_or_path,
            cache_dir = args.cache_dir,
            use_fast=True,
            local_files_only = True,
        )
        self.pretrain_model = T5ForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=args.model_name_or_path,
                cache_dir = args.cache_dir, # 就像是你下载东西要指定一个存放下载文件的文件夹一样，这里是指定模型相关缓存文件的存放位置。
                local_files_only = True, # 只从本地文件加载模型，不会尝试从网络上下载。
        )

        self.config:T5Config = self.pretrain_model.config

        if isinstance(self.pretrain_model, T5ForConditionalGeneration):
            self.match_n_layer = self.config.num_decoder_layers
            self.match_n_head = self.config.num_heads
            self.n_embd = self.config.d_model
            self.match_n_embd = self.config.d_kv
        else:
            raise ValueError("Other models are not supported yet!")


        # initial task-specific prompt
        if args.use_task:
            pass



        # # initial knowledge prompt
        if args.use_knowledge:
            pass


        self.dropout = nn.Dropout(args.prompt_dropout)




    def freeze_parameter(self):
        pass






    def init_task_prompt(self):
        pass




    def get_prompt(self):
        pass



    def forward(self):
        pass
