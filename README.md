# MPrompt-reproduce
---
Reproduce a prompt-learning technique: MPrompt, from the paper 《MPrompt: Exploring Multi-level Prompt Tuning for Machine Reading Comprehension》 
![image](https://github.com/user-attachments/assets/bd14077c-a150-4518-af31-5125a9ee5c6c)



## MPrompt
---
### Core Idea
MPrompt是一个多层次的提示调优方法,主要用于机器阅读理解任务。它通过**三个层次的提示**来增强预训练语言模型对输入语义的理解:

- 任务特定提示`(Task-specific Prompt)`
- 领域特定提示`(Domain-specific Prompt)`
- 上下文特定提示`(Context-specific Prompt)`

### 三层提示的具体设计
#### 任务特定提示:
- 与输入无关的提示,针对特定任务共享相同的提示信息
- 为预训练语言模型中不同类型的注意力层添加前缀

#### 领域特定提示:
- 利用数据集生成的领域知识
- 引入独立性约束(Independence Constraint),使每个提示专注于域内信息
- 避免**跨域知识的冗余**

#### 上下文特定提示:
- 依赖于输入的具体上下文
- 通过**提示生成器**为不同上下文生成不同的提示
- 提供比领域特定提示更细粒度的知识


### 提示生成器设计(Prompt Generator)
提示生成器是一个小规模的预训练语言模型,主要功能:
- 编码上下文信息
- 将上下文相关知识整合到提示生成过程中
- 确保为不同上下文生成不同的提示
- 增强提示的上下文感知能力

### 实验结果
#### 主要优势:
- 多层次提示设计提供了不同粒度的语义理解
- 独立性约束避免了信息冗余
- 提示生成器增强了上下文相关性

#### 实验效果:
- 在12个基准数据集上平均提升`1.94%`
- 相比其他软提示方法(如Prefix-tuning)有显著提升
- 在某些任务上甚至超过了`full-fine-tune`的性能




## Env config
```bash
pip install -r requirement.txt
```




## Training






## Citation
