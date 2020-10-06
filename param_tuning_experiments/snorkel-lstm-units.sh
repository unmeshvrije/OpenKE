#!/bin/sh

P=$1
source ~/.bashrc
# Result Directory
RD="/var/scratch2/xxx/OpenKE-results/"
E="complex"
DB="fb15k237"
#"fb15k237"
RDB=$RD"$DB/"
RDE=$RDB"embeddings/" # contains file in the name format : db-model.json
RDD=$RDB"data/" # contains files in the name format fb15k237-transe-training-topk-10.pkl (embedding features) or .json (raw answers)
RDM=$RDB"models/"
RDS=$RDB"subgraphs/" # contains files in the name format fb15k237-transe-{subgraphs/avgemb/varemb}-tau-10.pkl

K=10
tf=$RDD"$DB-$E-test-topk-$K.pkl"
to=$RDB"out/$DB-$E-annotated-topk-10-$P.out"

if [ "$2" = "a" ];
then
    python snorkel_lstm_units.py --testfile $tf --topk $K --db $DB --pred $P --true-out $to --abstain
else
    python snorkel_lstm_units.py --testfile $tf --topk $K --db $DB --pred $P --true-out $to
fi
