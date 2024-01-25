#!/bin/sh
source ~/.bashrc

python embedding_model.py --gpu --db $1 --mode "test" --model $2 --topk $3 -r $4
