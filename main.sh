#!/bin/sh
source ~/.bashrc

RD="/var/scratch2/uji300/OpenKE-results/"

for DB in "fb15k237" "dbpedia50"
do
    for M in "transe" "rotate" "complex"
    do
        if [ "$M" != "complex" ];
            ./step1-train-embedding-models.sh $RD $M $DB
        fi
        ./step2-create-subgraphs.sh $RD $M $DB
        ./step3-gen-input-for-lstm-mlp-classifiers.sh $RD $M $DB
        ./step4-train-lstm-mlp-classifiers.sh $RD $M $DB
        ./step5-generate-classifier-labels.sh $RD $M $DB
        for P in "head" "tail"
        do
            ./step6-compare-all-classifiers.sh $RD $M $DB $P 10 "a"
        done;
    done;
done;

