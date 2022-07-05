#!/bin/bash
cd /seu_share/home/dongfang/df_yindh/RSTT
module load anaconda3
module load cuda-11.6
conda init --all
conda create --yes --name venv
conda activate venv
conda install --yes --file requirements.conda.txt -c pytorch -c conda-forge -c anaconda