#!/bin/sh
rm -rf venv
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116