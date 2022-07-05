#!/bin/bash
#BSUB -J datasets #任务名称
#BSUB -n 1,28
#BSUB -m c06n13
#BSUB -q normal  #队列名称，可用bqueues查看
#BSUB -o /seu_share/home/dongfang/df_yindh/datasets.out
#BSUB -e /seu_share/home/dongfang/df_yindh/datasets.err
unzip -n /seu_share/home/dongfang/df_yindh/vimeo_septuplet.zip -d /seu_share/home/dongfang/df_yindh/RSTT/vimeo90k/
