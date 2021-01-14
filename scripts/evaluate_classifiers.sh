#!/bin/sh

cd ..
source venv/bin/activate

CLASSIFIERS="mlp_multi"
BASEDIR="/Users/jacopo/Desktop/binary-embeddings/"
DATASET="fb15k237"
OUTDIR="${BASEDIR}/${DATASET}/results/"
TOPK=10
MODEL="transe"
EXEC_CREATE_ANS="create_answer_annotations_classifier.py"
EXEC_EVAL_GOLD="evaluate_annotations_gold_standard.py"

echo "Creating annotations "
for TYPPRED in "head" "tail"; do
  for CLASSIFIER in `echo $CLASSIFIERS`; do
    echo "Annotating the $TYPPRED answers with $CLASSIFIER"
    python3 $EXEC_CREATE_ANS --result_dir $BASEDIR --topk $TOPK --db $DATASET --mode test --model $MODEL --type_prediction $TYPPRED --classifier $CLASSIFIER
  done
done

echo "Evaluating the performance on the gold standard ..."
for TYPPRED in "head" "tail"; do
  for CLASSIFIER in `echo $CLASSIFIERS`; do
    echo "Testing the gold standard with the $TYPPRED answers with $CLASSIFIER"
    python3 $EXEC_EVAL_GOLD --result_dir $BASEDIR --topk $TOPK --db $DATASET --mode test --model $MODEL --type_prediction $TYPPRED --classifier $CLASSIFIER
  done
done