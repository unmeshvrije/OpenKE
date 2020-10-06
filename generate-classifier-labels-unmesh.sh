#!/bin/sh
if [ "$#" -ne 2 ];
then
    echo "usage: $0 [model] [db]. Found $#"
    exit
fi

# Result Directory
#RD="experiments/"
#export PYTHONPATH="${PYTHONPATH}:./LibKGE/libkge"

RD="/var/scratch2/xxx/OpenKE-results/"
E=$1
DB=$2
#M="lstm"
#U=100
DR=0.2

RDB=$RD"$DB/"
RDE=$RDB"embeddings/" # contains file in the name format : db-model.json
RDD=$RDB"data/" # contains files in the name format fb15k237-transe-training-topk-10.pkl (embedding features) or .json (raw answers)
RDM=$RDB"models/"
RDS=$RDB"subgraphs/"

ENTDICT=$RD"$DB/misc/$DB-id-to-entity.pkl"
RELDICT=$RD"$DB/misc/$DB-id-to-relation.pkl"

for K in 3 # 1 5 # 10
do
    for P in  "head" # "tail"
    do
    for M in  "mlp" "lstm" # "path" "sub"
    do
    for U in 100 #200 10 #500
    do
        if [ $M  == "sub" ];
        then
            if [ $E == "complex" ];
            then
                emb_file=$RDE"$DB-$E.pt"
            else
                emb_file=$RDE"$DB-$E.json"
            fi
            sub_file=$RDS"$DB-$E-subgraphs-tau-10.pkl"
            sub_emb_file=$RDS"$DB-$E-avgemb-tau-10.pkl"
            test_file=$RDD"$DB-$E-test-topk-$K.pkl"
            python3 generate_classifier_labels.py --classifier $M --testfile $test_file --embfile $emb_file --subfile $sub_file --subembfile $sub_emb_file --topk $K --db $DB --pred $P --trainfile "./benchmarks/$DB/train2id.txt" --model $E -stp 0.01 --entdict $ENTDICT --reldict $RELDICT -rd $RD
        elif [ $M == "path" ];
        then
            if [ $E == "complex" ];
            then
                emb_file=$RDE"$DB-$E.pt"
            else
                emb_file=$RDE"$DB-$E.json"
            fi
            echo "$emb_file"
            test_file=$RDD"$DB-$E-test-topk-$K.pkl"
            python3 generate_classifier_labels.py --classifier $M --testfile $test_file --embfile $emb_file --topk $K --model $E --db $DB --pred $P --trainfile "./benchmarks/$DB/train2id.txt" -stp 0.01 --entdict $ENTDICT --reldict $RELDICT -rd $RD
        else
            mo_file=$RDM"$DB-$E-training-topk-10-$P-model-$M-units-$U-dropout-$DR.json"
            train_file=$RDD"$DB-$E-training-topk-$K.pkl"
            test_file=$RDD"$DB-$E-test-topk-$K.pkl"
            if [ ! -f  $mo_file ];
            then
                echo "$mo_file not found. Generating one...";
                python3 train_answer_model.py --infile $train_file --topk $K --mode train  --pred $P --db $DB --units $U --dropout $DR --model $M -rd $RD
                echo "DONE"
            else
                echo "$mo_file FOUND";
            fi
            echo "Test file : $test_file "
            wt_file=$RDM"$DB-$E-training-topk-10-$P-weights-$M-units-$U-dropout-$DR.h5"

            #fb15k237-rotate-training-topk-10-ju-tail-model-lstm-units-100-dropout-0.2-taulow-0.2-tauhigh-0.75.out
            for TAU_LOW in 0.2
            do
            for TAU_HIGH in 0.6
            do
            out_file=$RDB"out/$DB-$E-training-topk-$K-$P-weights-$M-units-$U-dropout-$U-taulow-$TAU_LOW-tauhigh-$TAU_HIGH.out"
            if [ ! -f $out_file ];
            then
                python3 generate_classifier_labels.py --classifier $M --testfile $test_file --modelfile $mo_file --weightsfile $wt_file --topk $K --db $DB --pred $P --entdict $ENTDICT --reldict $RELDICT -rd $RD -tl $TAU_LOW -th $TAU_HIGH
            fi
            done;
            done;
        fi
    done;
    done;
done;
done;
