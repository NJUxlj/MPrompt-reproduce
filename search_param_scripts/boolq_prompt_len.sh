#!/bin/bash  
MY_PYTHON="the path of python" # ������conda�о��廷���µ�python����·������
# ģ�Ͳ���
model_name='unifiedqa-t5-base'
model_name_or_path='allenai/unifiedqa-t5-base'
# ���ݼ�
data_dir="./qa_datasets"
dataset_name='boolq'
# �����ַ
output_dir="./search_output_dir/prompt_token"
max_debug_samples=0 #  ������

EPOCH=10
batch_size=8
val_batch_size=8
warmup_ratio=0.1 # warm up
gradient_accumulation_steps=1  # ԭ�ĵ�batch size���ر�С
step_log=$(( 30 * $gradient_accumulation_steps ))  # ԭ�������� �����ݶ��ۼ�
max_ques_length=512  # ques_opt �ĳ���
max_cont_length=512  # context �ĳ���
max_ans_length=10  # answer �ĳ���

# ���������ز���
num_beams=2

# prompt ����
use_task=True
use_domain=True
use_knowledge=True
ques_cont=True  # �����Ƿ����cont
prompt_dropout=0.1
# task����
task_sequence_length=10
lr=5e-5
init_task='random'
# knowledge ����
knowledge_lr=$lr  # �����һ��
# knowledge_sequence_length=15
map_hidden=True  # mlpӳ��hidden state,  False����self past key values
kd_prompt_dropout=0.1
# domain����
# n_prompt_tokens=5
domain_lr=$lr
gap=5
domain_size=3
domain_type=kmeans_context_3
loss_sample_n=3
domain_same_init=same  # prompt token�Ƿ�ͬ���ĳ�ʼ��  ['same', 'each_same', 'diff']  
use_enc_dec=False  # һֱ��False
domain_weight=0.0001
domain_loss_name=cka  # kl mmd cka None ������Լ��
cka_dynamic_weight=False # ��cka�Ƿ���ö�̬ѧϰ��
gap_knowledge=False  # domain prompt��ʼ���Ƿ�����knowledge  ΪTrue������

EXEC=./tdk/run_tdk.py   # ��������ʵ�����е�python�ļ���

export CUDA_VISIBLE_DEVICES="0"

# ��Ҫ�仯�Ĳ�����������prompt
prompt_lenght='5 10 15 20 30 40 50 60' #   

for knowledge_sequence_length in $prompt_lenght  #�˴��Ͳ�ʹ�ô������ˡ�
    do
        for n_prompt_tokens in $prompt_lenght
            do
                $MY_PYTHON $EXEC \
                --epoch $EPOCH \
                --batch_size $batch_size \
                --val_batch_size $val_batch_size \
                --lr $lr \
                --knowledge_lr $knowledge_lr \
                --data_dir $data_dir \
                --dataset_name $dataset_name \
                --step_log $step_log \
                --max_debug_samples $max_debug_samples \
                --output_dir $output_dir \
                --model_name $model_name \
                --model_name_or_path $model_name_or_path \
                --num_beams $num_beams \
                --gradient_accumulation_steps $gradient_accumulation_steps \
                --warmup_ratio $warmup_ratio \
                --use_task $use_task \
                --use_domain $use_domain \
                --use_knowledge $use_knowledge \
                --task_sequence_length $task_sequence_length \
                --knowledge_sequence_length $knowledge_sequence_length \
                --max_ques_length $max_ques_length \
                --max_cont_length $max_cont_length \
                --max_ans_length $max_ans_length \
                --ques_cont $ques_cont \
                --prompt_dropout $prompt_dropout \
                --domain_size $domain_size \
                --n_prompt_tokens $n_prompt_tokens \
                --domain_lr $domain_lr \
                --gap $gap \
                --domain_type $domain_type \
                --domain_same_init $domain_same_init \
                --loss_sample_n $loss_sample_n \
                --use_enc_dec $use_enc_dec \
                --domain_weight $domain_weight \
                --domain_loss_name $domain_loss_name \
                --cka_dynamic_weight $cka_dynamic_weight \
                --gap_knowledge $gap_knowledge \
                --kd_prompt_dropout $kd_prompt_dropout \
                --init_task $init_task \

        done
done