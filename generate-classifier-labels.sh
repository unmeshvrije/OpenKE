#!/bin/sh
if [ "$#" -ne 2 ];
then
    echo "usage: $0 [model] [db]. Found $#"
    exit
fi

# Result Directory
RD="experiments/"
export PYTHONPATH="${PYTHONPATH}:./LibKGE/libkge"

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

for K in 3 # 1 5
do
    for P in  "head" #"tail"
    do
    for M in  "mlp" "lstm" #"sub" "path"
    do
    for U in 100 #200 10 #500
    do
        if [ $M  == "sub" ];
        then
            sub_file=$RDS"$DB-$E-subgraphs-tau-10.pkl"
            mo_file=$RDB"out/$DB-$E-subgraphs-tau-10-$P.out"
            if [ -f  $mo_file ];
            then
              echo "$mo_file FOUND";
            else
              echo "$mo_file NOT FOUND";
              if [ $E == "complex" ];
              then
                  emb_file=$RDE"$DB-$E.pt"
              else
                  emb_file=$RDE"$DB-$E.json"
              fi
              sub_emb_file=$RDS"$DB-$E-avgemb-tau-10.pkl"
              test_file=$RDD"$DB-$E-test-topk-$K.pkl"
              python3 generate_classifier_labels.py -rd $RD --classifier $M --testfile $test_file --embfile $emb_file --subfile $sub_file --subembfile $sub_emb_file --topk $K --db $DB --pred $P --trainfile "./benchmarks/$DB/train2id.txt" --model $E -stp 0.01 --entdict $ENTDICT --reldict $RELDICT
            fi
        elif [ $M == "path" ];
        then
            mo_file=$RDB"out/$DB-$E-path-classifier-$P.out"
            if [ -f  $mo_file ];
            then
              echo "$mo_file FOUND";
            else
              echo "$mo_file NOT FOUND";
              if [ $E == "complex" ];
              then
                  emb_file=$RDE"$DB-$E.pt"
              else
                  emb_file=$RDE"$DB-$E.json"
              fi
              test_file=$RDD"$DB-$E-test-topk-$K.pkl"
              python3 generate_classifier_labels.py -rd $RD --classifier $M --testfile $test_file --embfile $emb_file --topk $K --model $E --db $DB --pred $P --trainfile "./benchmarks/$DB/train2id.txt" -stp 0.01 --entdict $ENTDICT --reldict $RELDICT
            fi
        else
            mo_file=$RDM"$DB-$E-training-topk-$K-ju-$P-model-$M-units-$U-dropout-$DR.json"
            train_file=$RDD"$DB-$E-training-topk-$K-ju.pkl"
            test_file=$RDD"$DB-$E-test-topk-$K-ju.pkl"
            if [ ! -f  $mo_file ];
            then
                echo "$mo_file not found. Generating one...";
                python3 train_answer_model.py --infile $train_file --topk $K --mode train  --pred $P --db $DB --units $U --dropout $DR --model $M -rd $RD
                echo "DONE"
            fi
            echo "$mo_file FOUND";
            echo "Test file : $test_file "
            wt_file=$RDM"$DB-$E-training-topk-$K-ju-$P-weights-$M-units-$U-dropout-$DR.h5"
            python3 generate_classifier_labels.py --classifier $M --testfile $test_file --modelfile $mo_file --weightsfile $wt_file --topk $K --db $DB --pred $P --entdict $ENTDICT --reldict $RELDICT -rd $RD -tl 0.25 -th 0.5
        fi
    done;
    done;
    done;
done;
