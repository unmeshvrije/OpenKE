#!/bin/sh

if [ "$#" -lt 3 ];
then
    echo "usage: $0 [transe/complex/rotate] [db] [head/tail]"
    exit
fi

source ~/.bashrc
# Result Directory
RD="/var/scratch2/xxx/OpenKE-results/"
E=$1
DB=$2
P=$3
K=10
#"fb15k237"
RDB=$RD"$DB/"
RDE=$RDB"embeddings/" # contains file in the name format : db-model.json
RDD=$RDB"data/" # contains files in the name format fb15k237-transe-training-topk-10.pkl (embedding features) or .json (raw answers)
RDM=$RDB"models/"
RDS=$RDB"subgraphs/" # contains files in the name format fb15k237-transe-{subgraphs/avgemb/varemb}-tau-10.pkl

#"transe"
tf=$RDD"$DB-$E-test-topk-$K.pkl"
to=$RDB"out/$DB-$E-annotated-topk-10-$P.out"
lo=$RDB"out/$DB-$E-training-topk-$K-$P-model-lstm-units-100-dropout-0.2.out"
mo=$RDB"out/$DB-$E-training-topk-$K-$P-model-mlp-units-100-dropout-0.2.out"
so=$RDB"out/$DB-$E-subgraphs-tau-10-$P.out"
po=$RDB"out/$DB-$E-path-classifier-$P.out"
ent_file=$RDB"misc/$DB-id-to-entity.pkl"
rel_file=$RDB"misc/$DB-id-to-relation.pkl"

echo "python supervised_learning.py --testfile $tf --topk $K --db $DB --pred $P --lstm-out $lo --mlp-out $mo --path-out $po --sub-out $so --true-out $to"
python supervised_learning.py --testfile $tf --topk $K --db $DB --pred $P --lstm-out $lo --mlp-out $mo --path-out $po --sub-out $so --true-out $to --entdict "/var/scratch2/xxx/OpenKE-results/$DB/misc/$DB-id-to-entity.pkl" --reldict "/var/scratch2/xxx/OpenKE-results/$DB/misc/$DB-id-to-relation.pkl" --model $E


#U=100
#DR=0.2
#for M in "lstm" #"mlp"
#do
#    mo_file=$RDM"$DB-$E-training-topk-$K-$P-model-$M-units-$U-dropout-$DR.json"
#    wt_file=$RDM"$DB-$E-training-topk-$K-$P-weights-$M-units-$U-dropout-$DR.h5"
#    python test_classifier.py --classifier $M"s" --testfile $tf --modelfile $mo_file --weightsfile $wt_file --topk $K --db $DB --pred $P --trainfile  $RDD"$DB-$E-training-topk-$K.pkl" --entdict $ent_file --reldict $rel_file
#done;
