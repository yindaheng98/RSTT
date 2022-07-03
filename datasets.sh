#!/bin/sh
curl -o vimeo90k/vimeo_septuplet.zip http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip
unzip -n ~/vimeo_septuplet.zip -d ~/RSTT/vimeo90k/
cp datasets/vimeo_septuplet/*.txt vimeo90k/vimeo_septuplet
python ./datasets/prepare_vimeo.py --path vimeo90k/vimeo_septuplet