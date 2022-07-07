#!/bin/bash
cd /seu_share/home/dongfang/df_yindh/RSTT
for p in experiments/RSTT-S/models/*.pth; do
p=$(echo $p | sed 's_/_\\/_g')
sed "s/<pretrain_model>/$p/g" < eval_seq_vimeo90k-bdcc.sh | bsub
done