#!/bin/sh

if [ "$#" -ne 3 ];
then
    echo "usage: $0 [result_dir] [model] [db]. Found $#"
    exit
fi

# Result Directory
RD=$1 #"/var/scratch2/uji300/OpenKE-results/"
E=$2
DB=$3
RDB=$RD"$DB/"
RDE=$RDB"embeddings/" # contains file in the name format : db-model.json
RDD=$RDB"data/" # contains files in the name format fb15k237-transe-training-topk-10.pkl (embedding features) or .json (raw answers)

if [ $E == "complex" ];
then
  embfile=$RDE"$DB-$E.pt"
else
  embfile=$RDE"$DB-$E.json"
fi
echo "$embfile"

# Generate training and test data for LSTM / MLP
for K in 10 #1 3 5
do
    tr_file=$RDD"$DB-$E-training-topk-$K.pkl"
    if [ ! -f  $tr_file ];
    then
        echo "$tr_file not found";
        python read_answers_and_pickle_numpy.py --embfile $embfile --db $DB --topk $K --mode train --ansfile $RDD"$DB-$E-training-topk-$K.json"
    else
        echo "$tr_file FOUND";
    fi

    te_file=$RDD"$DB-$E-test-topk-$K.pkl"
    if [ ! -f  $te_file ];
    then
        echo "$te_file not found";
        python read_answers_and_pickle_numpy.py --embfile $embfile --db $DB --topk $K --mode test --ansfile $RDD"$DB-$E-test-topk-$K.json"
    else
        echo "$te_file FOUND";
    fi
done;

