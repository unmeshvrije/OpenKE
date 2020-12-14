#!/bin/sh

if [ "$#" -lt 4 ];
then
echo "usage: $0 -m model -d db -r RecordsToTest -k [-1, -2, 10]. Found $#"
exit
fi

for (( i=1; i<=$#; i++ ))
do
    arg=${@:$i:1}   # Gets the string i
    val=${@:$i+1:1} # Gets the string i+1

    case $arg in
    -m)
        E=$val
        ((i++))
        ;;
    --model=*)
        # Split at = char and remove the shortest match from beginning
        E=${arg#*=}
        ;;
    -d)
        DB=$val
        ((i++))
        ;;
    --db=*)
        DB=${arg#*=}
        ;;
    -r)
        R=$val
        ((i++))
        ;;
    --record-test=*)
        R=${arg#*=}
        ;;
    -k)
        K=$val
        ((i++))
        ;;
        *)
        echo "Unknown argument number $i: '$arg'"
        ;;
    esac
done



# Result Directory
RD="/var/scratch2/uji300/OpenKE-results/"

#for E in "transe" "rotate" "complex"
#do
#for DB in "fb15k237" "dbpedia50"
#do

    RDB=$RD"$DB/"
    RDE=$RDB"embeddings/" # contains file in the name format : db-model.json
    RDD=$RDB"data/" # contains files in the name format fb15k237-transe-training-topk-10.pkl (embedding features) or .json (raw answers)
    RDM=$RDB"models/"
    RDS=$RDB"subgraphs/"

#for K in -2 #10 -1 # is for dynamic K, -2 is for dynamic threshold
#do
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
    edict_file="/var/scratch2/uji300/OpenKE-results/$DB/misc/$DB-id-to-entity.pkl"
    rdict_file="/var/scratch2/uji300/OpenKE-results/$DB/misc/$DB-id-to-relation.pkl"
    echo "Calling Python script"
    python test_subgraphs.py --testfile $test_file --embfile $emb_file --subfile $sub_file --subembfile $sub_emb_file --topk $K --db $DB --trainfile $train_file --model $E -stp 0.01 --entdict $edict_file --reldict $rdict_file --testonly $R
