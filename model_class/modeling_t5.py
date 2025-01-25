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


from transformers.modeling_outputs import (
    BaseModelOutput,
)


from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from transformers.models.t5.configuration_t5 import T5Config


logger = logging.get_logger(__name__)



