#!/bin/sh
if [ "$#" -ne 2 ];
then
    echo "usage: $0 [model] [db]. Found $#"
    exit
fi
# Result Directory
RD="/var/scratch2/uji300/OpenKE-results/"

E=$1
DB=$2
RDB=$RD"$DB/"
# Result Directory Embeddings
RDE=$RDB"embeddings/" # contains file in the name format : db-model.json

RDD=$RDB"data/" # contains files in the name format fb15k237-transe-training-topk-10.pkl (embedding features) or .json (raw answers)
RDM=$RDB"models/"
RDS=$RDB"subgraphs/"
U=100
DR=0.2
K=10

for P in  "head" # "tail"
do
    for M in "mlp" "lstm"
    do
        mo_file=$RDM"$DB-$E-training-topk-$K-$P-model-$M-units-$U-dropout-$DR.json"
        if [ ! -f  $mo_file ];
        then
            echo "$mo_file not found. Generating one...";
            python train_answer_model.py --infile $RDD"$DB-$E-training-topk-$K.pkl" --topk $K --mode train  --pred $P --db $DB --units $U --dropout $DR --model $M
            echo "DONE"
        else
            echo "$mo_file FOUND";
            wt_file=$RDM"$DB-$E-training-topk-$K-$P-weights-$M-units-$U-dropout-$DR.h5"
                python grid_search_lstm.py --classifier $M --testfile $RDD"$DB-$E-test-topk-$K.pkl" --modelfile $mo_file --weightsfile $wt_file --topk $K --db $DB --pred $P --abs-low 0.2 --abs-high 0.75 --true-out "/var/scratch2/uji300/OpenKE-results/$DB/out/$DB-$E-annotated-topk-10-$P.out"
        fi
    done;
done;
