#!/bin/sh
source ~/.bashrc
# Result Directory
DB="fb15k237"
RD="/var/scratch2/uji300/OpenKE-results/"
RDB=$RD"$DB/"
RDE=$RDB"embeddings/" # contains file in the name format : db-model.json
RDD=$RDB"data/" # contains files in the name format fb15k237-transe-training-topk-10.pkl (embedding features) or .json (raw answers)
RDM=$RDB"models/"
RDS=$RDB"subgraphs/" # contains files in the name format fb15k237-transe-{subgraphs/avgemb/varemb}-tau-10.pkl

E=$1 # "transe", "hole", "rotate"
K=10

# /var/scratch2/uji300/OpenKE-results//data/fb15k237-hole-training-topk-10.json
echo "$RDD"

echo "****"

python read_answers_and_log.py --ansfile "$RDD""$DB-$E-test-topk-$K.json" --mode test
#python read_answers_and_log.py --ansfile "$RDD""$DB-$E-training-topk-$K.json" --mode train
