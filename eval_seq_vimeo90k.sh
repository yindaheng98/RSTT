#!/bin/bash
for p in experiments/RSTT-S_archived_220705-001733/models/*.pth; do
echo "python eval_seq_vimeo90k.py --config ./configs/RSTT-S-eval-vimeo90k.yml --pretrain_model $p"
python eval_seq_vimeo90k.py --config ./configs/RSTT-S-eval-vimeo90k.yml --pretrain_model $p > $p.log 2>&1
done