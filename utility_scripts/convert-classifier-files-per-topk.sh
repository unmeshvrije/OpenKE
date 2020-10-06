#!/bin/sh

if [ "$#" -lt 2 ];
then
    echo "usage: $0 {transe/complex/rotate} {db}"
    exit
fi

source ~/.bashrc
# Result Directory
RD="/var/scratch2/xxx/OpenKE-results/"
E=$1
DB=$2
K=10
#"fb15k237"
RDB=$RD"$DB/"
RDE=$RDB"embeddings/" # contains file in the name format : db-model.json
RDD=$RDB"data/" # contains files in the name format fb15k237-transe-training-topk-10.pkl (embedding features) or .json (raw answers)
RDM=$RDB"models/"
RDS=$RDB"subgraphs/" # contains files in the name format fb15k237-transe-{subgraphs/avgemb/varemb}-tau-10.pkl

for P in "head" "tail"
do
    lo=$RDB"out/$DB-$E-training-topk-$K-$P-model-lstm-units-100-dropout-0.2.out"
    mo=$RDB"out/$DB-$E-training-topk-$K-$P-model-mlp-units-100-dropout-0.2.out"
    so=$RDB"out/$DB-$E-subgraphs-tau-10-$P-topk-$K.out"
    po=$RDB"out/$DB-$E-path-classifier-$P-topk-$K.out"

    python convert_classifier_files_per_topk.py $lo
    python convert_classifier_files_per_topk.py $mo
    python convert_classifier_files_per_topk.py $so
    python convert_classifier_files_per_topk.py $po
done
