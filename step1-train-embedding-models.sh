#!/bin/sh
source ~/.bashrc

if [ "$#" -ne 3 ];
then
    echo "usage: $0 [result_dir] [model] [db]. Found $#"
    exit
fi

RD=$1 #"/var/scratch2/uji300/OpenKE-results/"
E=$2  # "transe", "complex", "rotate"
DB=$3 #"fb15k237"
RDB=$RD"$DB/"
RDE=$RDB"embeddings/" # contains file in the name format : db-model.json
RDD=$RDB"data/" # contains files in the name format fb15k237-transe-training-topk-10.pkl (embedding features) or .json (raw answers)
RDM=$RDB"models/"
RDS=$RDB"subgraphs/" # contains files in the name format fb15k237-transe-{subgraphs/avgemb/varemb}-tau-10.pkl


EMB_MODEL_PATH="$RDE""$DB-$E.json"
echo "$EMB_MODEL_PATH"
if [ ! -f $EMB_MODEL_PATH ];
then
    python embedding_model.py --gpu --db $DB --mode "train" --model $E
fi

echo "Model $EMB_MODEL_PATH is found. Generating test answers..."
for K in 10 #1 3 5
do
    training_file=$RDD"$DB-$E-training-topk-$K.json"
    if [ ! -f $training_file ];
    then
        python embedding_model.py --gpu --db $DB --mode "trainAsTest" --model $E --topk $K
    fi

    test_file=$RDD"$DB-$E-test-topk-$K.json"
    if [ ! -f $test_file ];
    then
        python embedding_model.py --gpu --db $DB --mode "test" --model $E --topk $K
    fi
done
