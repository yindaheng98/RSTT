#!/bin/bash
#BSUB -J train-bdcc #任务名称
#BSUB -q gpu_v100  #队列名称，可用bqueues查看
#BSUB -gpu "num=1:mode=exclusive_process" #GPU数
#BSUB -o /seu_share/home/dongfang/df_yindh/train-bdcc.out
#BSUB -e /seu_share/home/dongfang/df_yindh/train-bdcc.err
cd /seu_share/home/dongfang/df_yindh/RSTT
module load anaconda3
module load cuda-11.6
conda activate venv
python train.py --config ./configs/RSTT-S.yml
