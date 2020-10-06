#!/bin/sh
if [ "$#" -ne 3 ];
then
    echo "usage: $0 [model] [db] [RecordsToTest]. Found $#"
    exit
fi

# Result Directory
RD="/var/scratch2/xxx/OpenKE-results/"

E=$1
DB=$2
R=$3
#for E in "transe" "rotate" "complex"
#do
#for DB in "fb15k237" "dbpedia50"
#do

RDB=$RD"$DB/"
RDE=$RDB"embeddings/" # contains file in the name format : db-model.json
RDD=$RDB"data/" # contains files in the name format fb15k237-transe-training-topk-10.pkl (embedding features) or .json (raw answers)
RDM=$RDB"models/"
RDS=$RDB"subgraphs/"

for K in -2 #10 -1 # is for dynamic K
do
    if [ $E == "complex" ];
    then
        emb_file=$RDE"$DB-$E.pt"
    else
        emb_file=$RDE"$DB-$E.json"
    fi
    sub_file=$RDS"$DB-$E-subgraphs-tau-10.pkl"
    sub_emb_file=$RDS"$DB-$E-avgemb-tau-10.pkl"
    test_file="./benchmarks/$DB/test2id.txt"
    train_file="./benchmarks/$DB/train2id.txt"
    edict_file="/var/scratch2/xxx/OpenKE-results/$DB/misc/$DB-id-to-entity.pkl"
    rdict_file="/var/scratch2/xxx/OpenKE-results/$DB/misc/$DB-id-to-relation.pkl"
    python test_subgraphs.py --testfile $test_file --embfile $emb_file --subfile $sub_file --subembfile $sub_emb_file --topk $K --db $DB --trainfile $train_file --model $E -stp 0.01 --entdict $edict_file --reldict $rdict_file --testonly $R
done;
