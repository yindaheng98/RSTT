#!/bin/sh
curl -o vimeo90k/vimeo_septuplet.zip http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip
cp datasets/vimeo_septuplet/*.txt vimeo90k/
python ./datasets/prepare_vimeo.py --path vimeo90k/