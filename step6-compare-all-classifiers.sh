#!/bin/sh

if [ "$#" -lt 5 ];
then
    echo "usage: $0 {result_dir} {transe/complex/rotate} {db} {head/tail} {topk} [a]"
    exit
fi

source ~/.bashrc
# Result Directory
RD=$1 #"/var/scratch2/uji300/OpenKE-results/"
E=$2
DB=$3
P=$4
K=$5
#"fb15k237"
RDB=$RD"$DB/"
RDE=$RDB"embeddings/" # contains file in the name format : db-model.json
RDD=$RDB"data/" # contains files in the name format fb15k237-transe-training-topk-10.pkl (embedding features) or .json (raw answers)
RDM=$RDB"models/"
RDS=$RDB"subgraphs/" # contains files in the name format fb15k237-transe-{subgraphs/avgemb/varemb}-tau-10.pkl

tf=$RDD"$DB-$E-test-topk-$K.pkl"
to=$RDB"out/$DB-$E-annotated-topk-$K-$P.out"
lo=$RDB"out/$DB-$E-training-topk-$K-$P-model-lstm-units-100-dropout-0.2.out" #-taulow-0.2-tauhigh-0.6.out"
mo=$RDB"out/$DB-$E-training-topk-$K-$P-model-mlp-units-100-dropout-0.2.out" #-taulow-0.2-tauhigh-0.6.out"
so=$RDB"out/$DB-$E-subgraphs-tau-10-$P-topk-$K.out"
po=$RDB"out/$DB-$E-path-classifier-$P-topk-$K.out"

if [ "$6" = "a" ];
then
    echo "python compare_all_classifiers.py --testfile $tf --topk $K --db $DB --pred $P --lstm-out $lo --mlp-out $mo --path-out $po --sub-out $so --true-out $to --abstain"

    sm="$RDM""$DB-$E-$P-snorkel.model.abs"
    if [ ! -f $sm ];
    then
        echo "$sm"
        echo "Snorkel label model not found. Running..."
        python run_and_save_snorkel_model.py --testfile $tf --topk $K --db $DB --pred $P --lstm-out $lo --mlp-out $mo --path-out $po --sub-out $so --true-out $to --abstain --model $E -rd $RD
    fi

    python compare_all_classifiers.py --testfile $tf --topk $K --db $DB --pred $P --lstm-out $lo --mlp-out $mo --path-out $po --sub-out $so --true-out $to --abstain --entdict "/var/scratch2/uji300/OpenKE-results/$DB/misc/$DB-id-to-entity.pkl" --reldict "/var/scratch2/uji300/OpenKE-results/$DB/misc/$DB-id-to-relation.pkl" --model $E -rd $RD

else
    echo "python compare_all_classifiers.py --testfile $tf --topk $K --db $DB --pred $P --lstm-out $lo --mlp-out $mo --path-out $po --sub-out $so --true-out $to"

    sm="$RDM""$DB-$E-$P-snorkel.model"
    if [ ! -f $sm ];
    then
        echo "$sm"
        echo "Snorkel label model not found. Running..."
        python run_and_save_snorkel_model.py --testfile $tf --topk $K --db $DB --pred $P --lstm-out $lo --mlp-out $mo --path-out $po --sub-out $so --true-out $to --model $E -rd $RD
    fi
    python compare_all_classifiers.py --testfile $tf --topk $K --db $DB --pred $P --lstm-out $lo --mlp-out $mo --path-out $po --sub-out $so --true-out $to --entdict "/var/scratch2/uji300/OpenKE-results/$DB/misc/$DB-id-to-entity.pkl" --reldict "/var/scratch2/uji300/OpenKE-results/$DB/misc/$DB-id-to-relation.pkl" --model $E -rd $RD
fi
