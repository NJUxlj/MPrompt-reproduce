# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model with prefix tuning. """


import warnings
import copy
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss


from transformers.activations import ACT2FN
from transformers.file_utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    replace_return_docstrings,
)

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)


from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.configuration_t5 import T5Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]



####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    # 加载TensorFlow模型权重到PyTorch模型中
    try:
        import re  # 正则表达式库，用于字符串匹配和处理
        import numpy as np  # 数值计算库
        import tensorflow as tf  # TensorFlow深度学习框架
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    
    # 获取TensorFlow检查点文件的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    
    # 从TF模型中加载权重
    # tf.train.list_variables() 返回检查点文件中所有变量的名称和形状
    init_vars = tf.train.list_variables(tf_path)
    names = []  # 存储变量名称
    tf_weights = {}  # 存储TensorFlow权重的字典
    
    # 遍历所有变量，加载权重数据
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # tf.train.load_variable() 从检查点文件中加载具体的变量值
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    # 处理每个权重变量
    for txt_name in names:
        # 将变量名按"/"分割，获取层级结构
        name = txt_name.split("/")
        
        # 跳过Adam优化器相关的变量，这些在预训练模型中不需要
        # adam_v, adam_m 是Adam优化器的动量参数
        # global_step 是训练步数计数器
        if any(
                n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
                for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        
        # 跳过包含"_slot_"的变量，这些通常是优化器的内部状态
        if "_slot_" in name[-1]:
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        
        # 开始权重映射过程
        pointer = model  # 指向当前PyTorch模型的指针
        array = tf_weights[txt_name]  # 获取对应的TensorFlow权重数组

        # 遍历变量名的每个部分，逐层定位到PyTorch模型中对应的参数
        for m_name in name:
            # 处理带数字后缀的层名（如layer_0, layer_1等）
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            
            # 根据TensorFlow的命名约定映射到PyTorch的参数名
            if scope_names[0] in ["kernel", "scale", "embedding"]:
                # TF中的kernel/scale/embedding对应PyTorch中的weight
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "self_attention":
                # 自注意力层，对应PyTorch中的第0层
                pointer = getattr(pointer, "layer")
                pointer = pointer[0]
            elif scope_names[0] == "enc_dec_attention":
                # 编码器-解码器注意力层，对应PyTorch中的第1层
                pointer = getattr(pointer, "layer")
                pointer = pointer[1]
            elif scope_names[0] == "dense_relu_dense":
                # 前馈网络层，对应PyTorch中的第2层
                pointer = getattr(pointer, "layer")
                pointer = pointer[2]
            elif scope_names[0] == "rms_norm":
                # RMS归一化层的映射
                if hasattr(pointer, "layer_norm"):
                    pointer = getattr(pointer, "layer_norm")
                elif hasattr(pointer, "final_layer_norm"):
                    pointer = getattr(pointer, "final_layer_norm")
            elif scope_names[0] == "scale":
                # scale参数对应weight
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                # 偏置参数的映射
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "squad":
                # SQuAD任务相关的分类器
                pointer = getattr(pointer, "classifier")
            elif scope_names[0] == "decoder" and name[1] == "logits":
                # 跳过解码器的logits层
                continue
            elif scope_names[0] == "logits":
                # logits层对应语言模型头
                pointer = getattr(pointer, "lm_head")
            elif scope_names[0] == "wi" and len(scope_names) > 1 and scope_names[1].isdigit():
                # 处理前馈网络中的wi_0, wi_1等权重
                pointer = getattr(pointer, f"wi_{scope_names[1]}")
                continue
            else:
                # 尝试直接通过属性名获取
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            
            # 如果有数字索引，进一步定位到具体的层
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        
        # 对于非kernel/scale/embedding的参数，需要获取其weight属性
        if scope_names[0] not in ["kernel", "scale", "embedding"]:
            pointer = getattr(pointer, "weight")
        
        # TensorFlow和PyTorch的权重矩阵通常需要转置
        # 因为两个框架的矩阵乘法约定不同
        if scope_names[0] != "embedding":
            logger.info(f"Transposing numpy weight of shape {array.shape} for {name}")
            array = np.transpose(array)
        
        # 验证权重形状是否匹配
        try:
            assert (
                    pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        
        # 将TensorFlow权重转换为PyTorch张量并赋值
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)  # 从字典中移除已处理的权重

    # 报告未被复制的权重
    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")
    return model




class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps


    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)


         # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states




class T5DenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)



    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = nn.functional.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)

        return hidden_states
    





class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()







class T5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseReluDense(config)
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedGeluDense(config)

        else:
            raise ValueError(
                f"{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )
        
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)

        return hidden_states







class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias



    def prune_heads(self, heads):
        pass
    

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """



    def compute_bias(self, query_length, key_length):
        pass



    def forward(
            self,
            hidden_states,
            mask=None,
            key_value_states=None,
            position_bias=None,
            past_key_value=None,
            layer_head_mask=None,
            query_length=None,
            use_cache=False,
            output_attentions=False,
            prefix=None,  # TODO: Chen
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """






class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        prefix=None,  # TODO: Chen
    ):
        
        normed_hidden_states = self.layer_norm(hidden_states)

        attention_output = self.SelfAttention.forward(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            prefix=prefix,  # TODO: Chen
        )

        hidden_states = hidden_states + self.dropout(attention_output[0])

        outputs =  (hidden_states,) + attention_output[1:]  # add attentions if we output them


        return outputs
    





class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
            self,
            hidden_states,
            key_value_states,
            attention_mask=None,
            position_bias=None,
            layer_head_mask=None,
            past_key_value=None,
            use_cache=False,
            query_length=None,
            output_attentions=False,
            prefix=None,  # TODO: Chen
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention.forward(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,   # 机制：在自回归解码时，存储之前生成的key和value，用于当前token的注意力计算
            use_cache=use_cache,
            query_length=query_length,   # 作用：在自回归解码时，限制解码器在生成当前token时只能看到前面的token
            output_attentions=output_attentions,
            prefix=prefix,  # TODO: Chen
        )
        # attention_output是一个元组，包含：
        # [0]: 注意力输出的隐藏状态 (batch_size, seq_len, d_model)
        # [1]: 如果use_cache=True，则包含当前层的key-value缓存用于下次解码
        # [2]: 如果output_attentions=True，则包含注意力权重矩阵 (batch_size, num_heads, seq_len, seq_len)
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs




class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))






