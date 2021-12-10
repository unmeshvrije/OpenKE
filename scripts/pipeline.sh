#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "Usage: pipeline.sh <BASE_DIR> <DATASET (e.g., fb15k237)> <MODEL (e.g., transe>"
    exit
fi

cd ..
#source venv/bin/activate

BASEDIR=$1
DATASET=$2
TOPK=10
MODEL=$3

OUTDIR="${BASEDIR}/${DATASET}/results/"
MODELDIR="${BASEDIR}/${DATASET}/models/"
ANNDIR="${BASEDIR}/${DATASET}/annotations/"
QUEDIR="${BASEDIR}/${DATASET}/queries/"
ANSDIR="${BASEDIR}/${DATASET}/answers/"
TRADIR="${BASEDIR}/${DATASET}/training_data/"
LOGDIR="${BASEDIR}/${DATASET}/logs/"

EXP=`date +"%Y-%m-%d-%T"`
EXPDIR="${LOGDIR}/$MODEL-$EXP"
SOUTFILE="${EXPDIR}/stdout"
SERRFILE="${EXPDIR}/stderr"

if [ ! -d "$OUTDIR" ]; then
  mkdir -p $OUTDIR
fi
if [ ! -d "$MODELDIR" ]; then
  mkdir -p $MODELDIR
fi
if [ ! -d "$ANNDIR" ]; then
  mkdir -p $ANNDIR
fi
if [ ! -d "$QUEDIR" ]; then
  mkdir -p $QUEDIR
fi
if [ ! -d "$ANSDIR" ]; then
  mkdir -p $ANSDIR
fi
if [ ! -d "$TRADIR" ]; then
  mkdir -p $TRADIR
fi
if [ ! -d "$LOGDIR" ]; then
  mkdir -p $LOGDIR
fi

# Store experimental results
if [ ! -d "$EXPDIR" ]; then
  mkdir -p $EXPDIR
fi
exec > >(tee -a $SOUTFILE)
exec 2> >(tee -a $ERRFILE)

echo "COMMAND: $0 $@"

EXEC_CREATE_QUERIES="create_queries.py"
EXEC_CREATE_ANSWERS="create_answers.py"
EXEC_CREATE_ANN_CWA="create_answer_annotations_cwa.py"
EXEC_CREATE_ANN_CLA="create_answer_annotations_classifier.py"
EXEC_EVAL_GOLD="evaluate_annotations_gold_standard.py"
EXEC_CREATE_TRAINING_DATA="create_training_data.py"
EXEC_CREATE_MODEL="create_model.py"

DO_STEP1="true"
DO_STEP2="true"
DO_STEP3="true"
DO_STEP4="true"
DO_STEP5="true"
DO_STEP6="true"
DO_STEP7="true"
DO_STEP8="true"
DO_STEP9="true"
DO_STEP10="true"
DO_STEP11="true"

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
        -SKIP1) DO_STEP1="false"
            ;;
        -SKIP2) DO_STEP2="false"
            ;;
        -SKIP3) DO_STEP3="false"
            ;;
        -SKIP4) DO_STEP4="false"
            ;;
        -SKIP5) DO_STEP5="false"
            ;;
        -SKIP6) DO_STEP6="false"
            ;;
        -SKIP7) DO_STEP7="false"
            ;;
        -SKIP8) DO_STEP8="false"
            ;;
        -SKIP9) DO_STEP9="false"
            ;;
        -SKIP10) DO_STEP10="false"
            ;;
        -SKIP11) DO_STEP11="false"
            ;;
    esac
    shift
done

source "scripts/pipeline-params-${DATASET}-${MODEL}.sh"

if [ $DO_STEP1 = "true" ]; then
# Step 1: Create a list of queries from the training data (train+valid) and test data
echo "1a- Create queries (train)"
python3 $EXEC_CREATE_QUERIES --result_dir $BASEDIR --db $DATASET --mode train
echo "1b- Create queries (test)"
python3 $EXEC_CREATE_QUERIES --result_dir $BASEDIR --db $DATASET --mode test
fi

if [ $DO_STEP2 = "true" ]; then
# Step 2: Create a list of answers for each query in the training and test datasets
echo "2a- Create answers (train)"
python3 $EXEC_CREATE_ANSWERS --mode train --type_prediction head --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANSWERS --mode train --type_prediction tail --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "2b- Create answers (test)"
python3 $EXEC_CREATE_ANSWERS --mode test --type_prediction head --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANSWERS --mode test --type_prediction tail --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
fi

if [ $DO_STEP3 = "true" ]; then
# Step 3: Annotate the answers of the training data using CWA
echo "3- Annotate the answers in the training dataset using CWA (to train the models)"
python3 $EXEC_CREATE_ANN_CWA --mode train --type_prediction head --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CWA --mode train --type_prediction tail --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
fi

if [ $DO_STEP4 = "true" ]; then
#Step 4: Create the training data for the classifiers
echo "4a- Create training data for the classifier MLP_MULTI"
python3 $EXEC_CREATE_TRAINING_DATA --type_prediction head --classifier mlp_multi --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_TRAINING_DATA --type_prediction tail --classifier mlp_multi --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "4b- Create training data for the classifier LSTM"
python3 $EXEC_CREATE_TRAINING_DATA --type_prediction head --classifier lstm --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_TRAINING_DATA --type_prediction tail --classifier lstm --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "4c- Create training data for the classifier CONV"
python3 $EXEC_CREATE_TRAINING_DATA --type_prediction head --classifier conv --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_TRAINING_DATA --type_prediction tail --classifier conv --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
fi

if [ $DO_STEP5 = "true" ]; then
#Step 5: Train the models
echo "5a- Create the model for the classifier MLP_MULTI"
python3 $EXEC_CREATE_MODEL $PARAMS_MLP_HEAD --type_prediction head --classifier mlp_multi --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_MODEL $PARAMS_MLP_TAIL --type_prediction tail --classifier mlp_multi --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "5b- Create the model for the classifier LSTM"
python3 $EXEC_CREATE_MODEL $PARAMS_LSTM_HEAD --type_prediction head --classifier lstm --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_MODEL $PARAMS_LSTM_TAIL --type_prediction tail --classifier lstm --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "5c- Create the model for the classifier CONV"
python3 $EXEC_CREATE_MODEL $PARAMS_CONV_HEAD --type_prediction head --classifier conv --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_MODEL $PARAMS_CONV_TAIL --type_prediction tail --classifier conv --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
fi

if [ $DO_STEP6 = "true" ]; then
#Step 6: Annotate the answers in the test dataset using the classifiers
echo "6a- Annotate the answers in the test dataset with the classifier MLP_MULTI"
python3 $EXEC_CREATE_ANN_CLA $PARAMS_MLP_HEAD --mode test --type_prediction head --classifier mlp_multi --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA $PARAMS_MLP_TAIL --mode test --type_prediction tail --classifier mlp_multi --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "6b- Annotate the answers in the test dataset with the classifier LSTM"
python3 $EXEC_CREATE_ANN_CLA $PARAMS_LSTM_HEAD --mode test --type_prediction head --classifier lstm --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA $PARAMS_LSTM_TAIL --mode test --type_prediction tail --classifier lstm --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "6c- Annotate the answers in the test dataset with the classifier CONV"
python3 $EXEC_CREATE_ANN_CLA $PARAMS_CONV_HEAD --mode test --type_prediction head --classifier conv --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA $PARAMS_CONV_TAIL --mode test --type_prediction tail --classifier conv --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "6d- Annotate the answers in the test dataset with the classifier PATH"
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction head --classifier path --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction tail --classifier path --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "6e- Annotate the answers in the test dataset with the classifier SUB"
python3 $EXEC_CREATE_ANN_CLA $PARAMS_SUB_HEAD --mode test --type_prediction head --classifier sub --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA $PARAMS_SUB_TAIL --mode test --type_prediction tail --classifier sub --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "6f- Annotate the answers in the test dataset with the classifier RANDOM"
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction head --classifier random --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction tail --classifier random --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "6g- Annotate the answers in the test dataset with the classifier THRESHOLD"
python3 $EXEC_CREATE_ANN_CLA $PARAMS_THRESHOLD_K_HEAD --mode test --type_prediction head --classifier threshold --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA $PARAMS_THRESHOLD_K_TAIL --mode test --type_prediction tail --classifier threshold --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
fi

if [ $DO_STEP7 = "true" ]; then
#Step 7: Repeat step 6 but this time using the answers in the training dataset (this data is needed to train snorkel)
echo "7a- Annotate the answers in the train dataset (for Snorkel) with the classifier MLP_MULTI"
python3 $EXEC_CREATE_ANN_CLA $PARAMS_MLP_HEAD --mode train --type_prediction head --classifier mlp_multi --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA $PARAMS_MLP_TAIL --mode train --type_prediction tail --classifier mlp_multi --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "7b- Annotate the answers in the train dataset (for Snorkel) with the classifier LSTM"
python3 $EXEC_CREATE_ANN_CLA $PARAMS_LSTM_HEAD --mode train --type_prediction head --classifier lstm --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA $PARAMS_LSTM_TAIL --mode train --type_prediction tail --classifier lstm --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "7c- Annotate the answers in the train dataset (for Snorkel) with the classifier CONV"
python3 $EXEC_CREATE_ANN_CLA $PARAMS_CONV_HEAD --mode train --type_prediction head --classifier conv --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA $PARAMS_CONV_TAIL --mode train --type_prediction tail --classifier conv --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "7d- Annotate the answers in the train dataset (for Snorkel) with the classifier PATH"
python3 $EXEC_CREATE_ANN_CLA --mode train --type_prediction head --classifier path --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode train --type_prediction tail --classifier path --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "7e- Annotate the answers in the train dataset (for Snorkel) with the classifier SUB"
python3 $EXEC_CREATE_ANN_CLA $PARAMS_SUB_HEAD --mode train --type_prediction head --classifier sub --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA $PARAMS_SUB_TAIL --mode train --type_prediction tail --classifier sub --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "7f- Annotate the answers in the train dataset (for Snorkel) with the classifier THRESHOLD"
python3 $EXEC_CREATE_ANN_CLA $PARAMS_THRESHOLD_K_HEAD --mode train --type_prediction head --classifier threshold --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA $PARAMS_THRESHOLD_K_TAIL --mode train --type_prediction tail --classifier threshold --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
fi

if [ $DO_STEP8 = "true" ]; then
#Step 8: Create the training dataset to train the Snorkel model and other ensemble models
echo "Step 8a: Create the training dataset to train the Snorkel model"
python3 $EXEC_CREATE_TRAINING_DATA $PARAMS_SNORKEL_HEAD --type_prediction head --classifier snorkel --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_TRAINING_DATA $PARAMS_SNORKEL_TAIL --type_prediction tail --classifier snorkel --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 8b: Create the training dataset to train the supensemble model"
python3 $EXEC_CREATE_TRAINING_DATA --type_prediction head --classifier supensemble --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_TRAINING_DATA --type_prediction tail --classifier supensemble --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 8c: Create the training dataset to train the SQUID model"
python3 $EXEC_CREATE_TRAINING_DATA $PARAMS_SQUID_HEAD --type_prediction head --classifier squid --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_TRAINING_DATA $PARAMS_SQUID_TAIL --type_prediction tail --classifier squid --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
fi

if [ $DO_STEP9 = "true" ]; then
#Step 9: Create the Snorkel and other ensemble models
echo "Step 9a: Create the Snorkel model"
python3 $EXEC_CREATE_MODEL $PARAMS_SNORKEL_HEAD --type_prediction head --classifier snorkel --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_MODEL $PARAMS_SNORKEL_TAIL --type_prediction tail --classifier snorkel --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 9b: Create the supensemble model"
python3 $EXEC_CREATE_MODEL --type_prediction head --classifier supensemble --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_MODEL --type_prediction tail --classifier supensemble --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 9c: Create the squid model"
python3 $EXEC_CREATE_MODEL $PARAMS_SQUID_HEAD --type_prediction head --classifier squid --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_MODEL $PARAMS_SQUID_TAIL --type_prediction tail --classifier squid --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
fi

if [ $DO_STEP10 = "true" ]; then
#Step 10: Annotate the answers using the ensemble models
echo "Step 10a: Annotate the answers in the test set using Snorkel"
python3 $EXEC_CREATE_ANN_CLA $PARAMS_SNORKEL_HEAD --mode test --type_prediction head --classifier snorkel --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA $PARAMS_SNORKEL_TAIL --mode test --type_prediction tail --classifier snorkel --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 10b: Annotate the answers in the test set using min"
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction head --classifier min --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction tail --classifier min --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 10c: Annotate the answers in the test set using maj"
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction head --classifier maj --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction tail --classifier maj --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 10d: Annotate the answers in the test set using supensemble"
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction head --classifier supensemble --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction tail --classifier supensemble --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 10e: Annotate the answers in the test set using squid"
python3 $EXEC_CREATE_ANN_CLA $PARAMS_SQUID_HEAD --mode test --type_prediction head --classifier squid --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA $PARAMS_SQUID_TAIL --mode test --type_prediction tail --classifier squid --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
fi

if [ $DO_STEP11 = "true" ]; then
#Step 11: Test the performance using the gold standard
echo "Step 11a: Test the performance using MLP_MULTI and the gold standard"
python3 $EXEC_EVAL_GOLD --mode test --type_prediction head --classifier mlp_multi --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_EVAL_GOLD --mode test --type_prediction tail --classifier mlp_multi --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 11b: Test the performance using LSTM and the gold standard"
python3 $EXEC_EVAL_GOLD --mode test --type_prediction head --classifier lstm --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_EVAL_GOLD --mode test --type_prediction tail --classifier lstm --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 11c: Test the performance using CONV and the gold standard"
python3 $EXEC_EVAL_GOLD --mode test --type_prediction head --classifier conv --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_EVAL_GOLD --mode test --type_prediction tail --classifier conv --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 11d: Test the performance using PATH and the gold standard"
python3 $EXEC_EVAL_GOLD --mode test --type_prediction head --classifier path --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_EVAL_GOLD --mode test --type_prediction tail --classifier path --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 11e: Test the performance using SUB and the gold standard"
python3 $EXEC_EVAL_GOLD --mode test --type_prediction head --classifier sub --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_EVAL_GOLD --mode test --type_prediction tail --classifier sub --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 11f: Test the performance using RANDOM and the gold standard"
python3 $EXEC_EVAL_GOLD --mode test --type_prediction head --classifier random --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_EVAL_GOLD --mode test --type_prediction tail --classifier random --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 11g: Test the performance using MIN and the gold standard"
python3 $EXEC_EVAL_GOLD --mode test --type_prediction head --classifier min --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_EVAL_GOLD --mode test --type_prediction tail --classifier min --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 11g: Test the performance using MAJ and the gold standard"
python3 $EXEC_EVAL_GOLD --mode test --type_prediction head --classifier maj --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_EVAL_GOLD --mode test --type_prediction tail --classifier maj --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 11h: Test the performance using THRESHOLD and the gold standard"
python3 $EXEC_EVAL_GOLD --mode test --type_prediction head --classifier threshold --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_EVAL_GOLD --mode test --type_prediction tail --classifier threshold --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 11i: Test the performance using SUPENSEMBLE and the gold standard"
python3 $EXEC_EVAL_GOLD --mode test --type_prediction head --classifier supensemble --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_EVAL_GOLD --mode test --type_prediction tail --classifier supensemble --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 11l: Test the performance using Snorkel and the gold standard"
python3 $EXEC_EVAL_GOLD --mode test --type_prediction head --classifier snorkel --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_EVAL_GOLD --mode test --type_prediction tail --classifier snorkel --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 11m: Test the performance using SQUID and the gold standard"
python3 $EXEC_EVAL_GOLD --mode test --type_prediction head --classifier squid --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_EVAL_GOLD --mode test --type_prediction tail --classifier squid --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
fi

# Store results
echo "Copying data ..."
cp scripts/$0 $EXPDIR
cp scripts/pipeline-params-${DATASET}-${MODEL}.sh $EXPDIR
mkdir -p $EXPDIR/results
cp -R $OUTDIR $EXPDIR/results
mkdir -p $EXPDIR/models
cp -R $MODELDIR/ $EXPDIR/models
mkdir -p $EXPDIR/source
cp -R *.py $EXPDIR/source/
mkdir -p $EXPDIR/source/support
cp -R support/ $EXPDIR/source/support
echo "done."
