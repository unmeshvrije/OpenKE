#!/bin/sh
source ~/.bashrc

if [ "$#" -ne 3 ];
then
    echo "usage: $0 [result_dir] [model] [db]. Found $#"
    exit
fi

RD=$1 #"/var/scratch2/uji300/OpenKE-results/"
E=$2  # "transe", "hole", "rotate"
DB=$3 #"fb15k237"
RDB=$RD"$DB/"
RDE=$RDB"embeddings/" # contains file in the name format : db-model.json
RDM=$RDB"models/"
RDS=$RDB"subgraphs/" # contains files in the name format fb15k237-transe-{subgraphs/avgemb/varemb}-tau-10.pkl

if [ $E == "complex" ];
then
  emb_file=$RDE"$DB-$E.pt"
else
  emb_file=$RDE"$DB-$E.json"
fi
in_file="./benchmarks/$DB/train2id.txt"

SUB_FILE_PATH="$RDS""$DB-$E-subgraphs-tau-10.pkl"
if [ ! -f $SUB_FILE_PATH ];
then
    echo "$SUB_FILE_PATH : NOT FOUND. Generating..."
    python create_subgraphs.py --db $DB --model $E --embfile $emb_file --ms 10 --infile $in_file
fi

