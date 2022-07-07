#!/bin/bash
#BSUB -J train-bdcc #任务名称
#BSUB -q gpu_v100  #队列名称，可用bqueues查看
#BSUB -gpu "num=1:mode=exclusive_process" #GPU数
#BSUB -o /seu_share/home/dongfang/df_yindh/RSTT/<pretrain_model>.out
#BSUB -e /seu_share/home/dongfang/df_yindh/RSTT/<pretrain_model>.err
cd /seu_share/home/dongfang/df_yindh/RSTT
module load anaconda3
module load cuda-11.6
eval "$(conda shell.bash hook)"
conda activate venv
python eval_seq_vimeo90k.py --config ./configs/RSTT-S-eval-vimeo90k.yml --pretrain_model <pretrain_model>
