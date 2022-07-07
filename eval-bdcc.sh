#!/bin/bash
cd /seu_share/home/dongfang/df_yindh/RSTT
for p in experiments/RSTT-S_archived_220705-162106/models/*.pth; do
p=$(echo $p | sed 's_/_\\/_g')
sed "s/<pretrain_model>/$p/g" < eval_seq_vimeo90k-bdcc.sh | bsub
done