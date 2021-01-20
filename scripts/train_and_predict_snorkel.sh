#!/bin/sh

cd ..
source venv/bin/activate

BASEDIR=$1
DATASET="fb15k237"
OUTDIR="${BASEDIR}/${DATASET}/results/"
TOPK=10
MODEL="transe"
EXEC_CREATE_ANS="create_answer_annotations_classifier.py"
EXEC_EVAL_GOLD="evaluate_annotations_gold_standard.py"
EXEC_CREATE_MODEL="create_model.py"
EXEC_CREATE_TRAINING_DATA="create_training_data.py"
SNORKEL_CLASSIFIERS="mlp_multi lstm conv"

# First prepare the training data by executing the various classifiers on "Training data"
for TYPPRED in "head" "tail"; do
  for CLASSIFIER in `echo $SNORKEL_CLASSIFIERS`; do
    echo "Annotating the $TYPPRED answers with $CLASSIFIER"
    python3 $EXEC_CREATE_ANS --result_dir $BASEDIR --topk $TOPK --db $DATASET --mode train --model $MODEL --type_prediction $TYPPRED --classifier $CLASSIFIER
  done
done

# Second, prepare the training data for snorkel
echo "Creating training data ..."
python3 $EXEC_CREATE_TRAINING_DATA --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL --type_prediction head --classifier snorkel
python3 $EXEC_CREATE_TRAINING_DATA --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL --type_prediction tail --classifier snorkel

# Third, train the model for Snorkel (no true labels are used)
echo "Training models ..."
python3 $EXEC_CREATE_MODEL --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL --type_prediction head --classifier snorkel
python3 $EXEC_CREATE_MODEL --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL --type_prediction tail --classifier snorkel

# Fourth, perform the predictions with snorkel
echo "Making predictions on the test set ..."
python3 $EXEC_CREATE_ANS --result_dir $BASEDIR --topk $TOPK --db $DATASET --mode test --model $MODEL --type_prediction head --classifier snorkel
python3 $EXEC_CREATE_ANS --result_dir $BASEDIR --topk $TOPK --db $DATASET --mode test --model $MODEL --type_prediction tail --classifier snorkel

# Fifth, evaluate the predictions vs. the gold standard
echo "Testing predictions vs the gold standard ..."
python3 $EXEC_EVAL_GOLD --result_dir $BASEDIR --topk $TOPK --db $DATASET --mode test --model $MODEL --type_prediction head --classifier snorkel
python3 $EXEC_EVAL_GOLD --result_dir $BASEDIR --topk $TOPK --db $DATASET --mode test --model $MODEL --type_prediction tail --classifier snorkel