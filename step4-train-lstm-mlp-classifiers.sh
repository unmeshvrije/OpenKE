#!/bin/sh
if [ "$#" -ne 3 ];
then
    echo "usage: $0 [result_dir] [model] [db]. Found $#"
    exit
fi

# Result Directory
RD=$1 # "/var/scratch2/uji300/OpenKE-results/"
E=$2
DB=$3
RDB=$RD"$DB/"
RDE=$RDB"embeddings/" # contains file in the name format : db-model.json
RDD=$RDB"data/" # contains files in the name format fb15k237-transe-training-topk-10.pkl (embedding features) or .json (raw answers)
RDM=$RDB"models/"
#E="transe"
M="lstm"
U=100
DR=0.2

for K in 10 # 1 3 5
do
    for P in "head" "tail"
    do
    for M in "lstm" "mlp"
    do
    for U in 100 #200 #500
    do
    for DR in 0.2 #0.5 0.8
    do
    mo_file=$RDM"$DB-$E-training-topk-$K-$P-model-$M-units-$U-dropout-$DR.json"
    if [ ! -f  $mo_file ];
    then
        echo "$mo_file not found. Generating one...";
        python train_answer_model.py --infile $RDD"$DB-$E-training-topk-$K.pkl" --topk $K --mode train  --pred $P --db $DB --units $U --dropout $DR --model $M
        echo "DONE"
    else
        echo "$mo_file FOUND";
    fi
    done;
    done;
    done;
    done;
done;
