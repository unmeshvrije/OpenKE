#!/bin/sh
if [ "$#" -ne 2 ];
then
    echo "usage: $0 [model] [db]. Found $#"
    exit
fi

# Result Directory
RD="/Users/jacopo/Desktop/data_unmesh/"

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

for K in 10
do
    for P in  "head" "tail"
    do
    for M in  "mlp" "lstm" "path" "sub"
    do
    for U in 100
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
            python3 test_classifier.py --classifier $M --testfile $test_file --embfile $emb_file --subfile $sub_file --subembfile $sub_emb_file --topk $K --db $DB --pred $P --trainfile "./benchmarks/$DB/train2id.txt" --model $E -stp 0.01 --entdict "/var/scratch2/uji300/OpenKE-results/$DB/misc/$DB-id-to-entity.pkl" --reldict "/var/scratch2/uji300/OpenKE-results/$DB/misc/$DB-id-to-relation.pkl"
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
            python3 test_classifier.py --classifier $M --testfile $test_file --embfile $emb_file --topk $K --model $E --db $DB --pred $P --trainfile "./benchmarks/$DB/train2id.txt" -stp 0.01 --entdict "/var/scratch2/uji300/OpenKE-results/$DB/misc/$DB-id-to-entity.pkl" --reldict "/var/scratch2/uji300/OpenKE-results/$DB/misc/$DB-id-to-relation.pkl"

        else
            mo_file=$RDM"$DB-$E-training-topk-$K-$P-model-$M-units-$U-dropout-$DR.json"
            if [ ! -f  $mo_file ];
            then
                echo "$mo_file not found. Generating one...";
                python3 train_answer_model.py --infile $RDD"$DB-$E-training-topk-$K-ju.pkl" --topk $K --mode train  --pred $P --db $DB --units $U --dropout $DR --model $M
                echo "DONE"
            else
                echo "$mo_file FOUND";
                wt_file=$RDM"$DB-$E-training-topk-$K-$P-weights-$M-units-$U-dropout-$DR.h5"
                if [ $K -ge 50 ];
                then
                    python3 test_classifier.py --classifier $M --testfile $RDB"batch_data/" --modelfile $mo_file --weightsfile $wt_file --topk $K --db $DB --pred $P
                else
                    python3 test_classifier.py --classifier $M --testfile $RDD"$DB-$E-test-topk-$K.pkl" --modelfile $mo_file --weightsfile $wt_file --topk $K --db $DB --pred $P --entdict "/var/scratch2/uji300/OpenKE-results/$DB/misc/$DB-id-to-entity.pkl" --reldict "/var/scratch2/uji300/OpenKE-results/$DB/misc/$DB-id-to-relation.pkl"

                fi
            fi
        fi
    done;
    done;
done;
done;
