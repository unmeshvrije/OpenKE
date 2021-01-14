#!/bin/sh

cd ..
source venv/bin/activate

BASEDIR="/Users/jacopo/Desktop/binary-embeddings/"
DATASET="fb15k237"
OUTDIR="${BASEDIR}/${DATASET}/results/"
TOPK=10
MODEL="transe"

EXEC_CREATE_QUERIES="create_queries.py"
EXEC_CREATE_ANSWERS="create_answers.py"
EXEC_CREATE_ANN_CWA="create_answer_annotations_cwa.py"
EXEC_CREATE_ANN_CLA="create_answer_annotations_classifier.py"
EXEC_EVAL_GOLD="evaluate_annotations_gold_standard.py"
EXEC_CREATE_TRAINING_DATA="create_training_data.py"
EXEC_CREATE_MODEL="create_model.py"

DO_STEP1="false"
DO_STEP2="false"
DO_STEP3="false"
DO_STEP4="false"
DO_STEP5="true"
DO_STEP6="false"
DO_STEP7="false"
DO_STEP8="false"
DO_STEP9="false"
DO_STEP10="false"
DO_STEP11="false"

PARAMS_SNORKEL="--name_signals mlp_multi,lstm,conv,path,sub --snorkel_low_threshold 0.3,0.3,0.3,0,0 --snorkel_high_threshold 0.7,0.7,0.7,0.5,0.5"
PARAMS_MLP_HEAD="--mlp_dropout 0.1 --mlp_n_hidden_units 100"
PARAMS_MLP_TAIL="--mlp_dropout 0.1 --mlp_n_hidden_units 100"
PARAMS_LSTM_HEAD="--mlp_dropout 0.1 --mlp_n_hidden_units 100"
PARAMS_LSTM_TAIL="--mlp_dropout 0.1 --mlp_n_hidden_units 100"

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
python3 $EXEC_CREATE_ANSWERS --mode train --type_prediction head --known_answers_train_file benchmarks/fb15k237/train2id.txt --known_answers_valid_file benchmarks/fb15k237/valid2id.txt --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANSWERS --mode train --type_prediction tail --known_answers_train_file benchmarks/fb15k237/train2id.txt --known_answers_valid_file benchmarks/fb15k237/valid2id.txt --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "2b- Create answers (test)"
python3 $EXEC_CREATE_ANSWERS --mode test --type_prediction head --known_answers_train_file benchmarks/fb15k237/train2id.txt --known_answers_valid_file benchmarks/fb15k237/valid2id.txt --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANSWERS --mode test --type_prediction tail --known_answers_train_file benchmarks/fb15k237/train2id.txt --known_answers_valid_file benchmarks/fb15k237/valid2id.txt --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
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
python3 $EXEC_CREATE_MODEL --type_prediction head --classifier conv --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_MODEL --type_prediction tail --classifier conv --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
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
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction head --classifier conv --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction tail --classifier conv --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "6d- Annotate the answers in the test dataset with the classifier PATH"
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction head --classifier path --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction tail --classifier path --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "6e- Annotate the answers in the test dataset with the classifier SUB"
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction head --classifier sub --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction tail --classifier sub --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "6f- Annotate the answers in the test dataset with the classifier RANDOM"
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction head --classifier random --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction tail --classifier random --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
fi

if [ $DO_STEP7 = "true" ]; then
#Step 7: Repeat step 6 but this time using the answers in the training dataset (this data is needed to train snorkel)
echo "7a- Annotate the answers in the train dataset (for Snorkel) with the classifier MLP_MULTI"
python3 $EXEC_CREATE_ANN_CLA --mode train --type_prediction head --classifier mlp_multi --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode train --type_prediction tail --classifier mlp_multi --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "7b- Annotate the answers in the train dataset (for Snorkel) with the classifier LSTM"
python3 $EXEC_CREATE_ANN_CLA --mode train --type_prediction head --classifier lstm --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode train --type_prediction tail --classifier lstm --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "7c- Annotate the answers in the train dataset (for Snorkel) with the classifier CONV"
python3 $EXEC_CREATE_ANN_CLA --mode train --type_prediction head --classifier conv --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode train --type_prediction tail --classifier conv --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "7d- Annotate the answers in the train dataset (for Snorkel) with the classifier PATH"
python3 $EXEC_CREATE_ANN_CLA --mode train --type_prediction head --classifier path --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode train --type_prediction tail --classifier path --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "7e- Annotate the answers in the train dataset (for Snorkel) with the classifier SUB"
python3 $EXEC_CREATE_ANN_CLA --mode train --type_prediction head --classifier sub --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode train --type_prediction tail --classifier sub --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "7f- Annotate the answers in the train dataset (for Snorkel) with the classifier RANDOM"
python3 $EXEC_CREATE_ANN_CLA --mode train --type_prediction head --classifier random --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode train --type_prediction tail --classifier random --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
fi

if [ $DO_STEP8 = "true" ]; then
#Step 8: Create the training dataset to train the Snorkel model
echo "Step 8: Create the training dataset to train the Snorkel model"
python3 $EXEC_CREATE_TRAINING_DATA $PARAMS_SNORKEL --type_prediction head --classifier snorkel --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_TRAINING_DATA $PARAMS_SNORKEL --type_prediction tail --classifier snorkel --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
fi

if [ $DO_STEP9 = "true" ]; then
#Step 9: Create the Snorkel model
echo "Step 9: Create the Snorkel model"
python3 $EXEC_CREATE_MODEL $PARAMS_SNORKEL --type_prediction head --classifier snorkel --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_MODEL $PARAMS_SNORKEL --type_prediction tail --classifier snorkel --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
fi

if [ $DO_STEP10 = "true" ]; then
#Step 10: Annotate the answers using the ensemble models
echo "Step 10a: Annotate the answers in the test set using Snorkel"
python3 $EXEC_CREATE_ANN_CLA $PARAMS_SNORKEL --mode test --type_prediction head --classifier snorkel --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA $PARAMS_SNORKEL --mode test --type_prediction tail --classifier snorkel --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 10b: Annotate the answers in the test set using min"
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction head --classifier min --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction tail --classifier min --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
echo "Step 10c: Annotate the answers in the test set using maj"
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction head --classifier maj --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_CREATE_ANN_CLA --mode test --type_prediction tail --classifier maj --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
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
echo "Step 11h: Test the performance using Snorkel and the gold standard"
python3 $EXEC_EVAL_GOLD --mode test --type_prediction head --classifier snorkel --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
python3 $EXEC_EVAL_GOLD --mode test --type_prediction tail --classifier snorkel --result_dir $BASEDIR --topk $TOPK --db $DATASET --model $MODEL
fi




