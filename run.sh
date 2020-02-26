cd openke
bash make.sh
cd ..
#python test_transe_WN18_adv_sigmoidloss.py gpu $1
#echo "Find 50 topk answers..."
#python train_transe_FB15K237.py gpu 10 "test"
#echo "Prepare training for LSTM..."
#python "read-answers.py" 10 "/home/uji300/OpenKE/FB15K237-results-scores-10-filtered.json"
#echo "Training for LSTM..."
## python program training_file mode topk head/tail predictions
#python lstm.py "/var/scratch2/uji300/OpenKE-results/fb15k237-training-topk-10.json" "train" 10 "tail"
#echo "Testing for LSTM..."
#python lstm.py "/var/scratch2/uji300/OpenKE-results/fb15k237-training-topk-10.json" "train" 10 "tail"
#python lstm.py "/var/scratch2/uji300/OpenKE-results/fb15k237-test-topk-10.json" "test" 10 "tail"


#python train_model.py --gpu --db fb15k237 --mode train --model complex


# /home/uji300/OpenKE/result/fb15k237-transe.json

#    outfile_name = result_dir + args.db + "-results-scores-"+args.mode+"-"+str(topk)+"-"+fil+".json"

DB="fb15k237"
RESULT_DIR="/var/scratch2/uji300/OpenKE-results/$DB/"

RESULT_PATH=$RESULT_DIR"result/$DB-transe.json"

if [ ! -f $RESULT_PATH ];
then
    python model.py --gpu --db "fb15k237" --mode "train" --model "transe"
fi

python model.py --gpu --db "fb15k237" --mode "test" --model "transe"
python model.py --gpu --db "fb15k237" --mode "test" --model "transe" --filtered
python model.py --gpu --db "fb15k237" --mode "trainAsTest" --model "transe"

python read-answers.py --embfile "$RESULT_DIR""result/$DB-transe.json" --topk 10 --db "fb15k237" --ansfile "$RESULT_DIR""fb15k237-transe-results-scores-trainAsTest-topk-10.json"
python read-answers.py --embfile "$RESULT_DIR""result/$DB-transe.json" --topk 10 --db "fb15k237" --ansfile "$RESULT_DIR""fb15k237-transe-results-scores-test-topk-10-filtered.json"
python read-answers.py --embfile "$RESULT_DIR""result/$DB-transe.json" --topk 10 --db "fb15k237" --ansfile "$RESULT_DIR""fb15k237-transe-results-scores-test-topk-10-unfiltered.json"


python lstm.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-trainAsTest-topk-10.json  --topk 10 --mode train --pred "head" --db fb15k237

python lstm.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-unfiltered.json --topk 10 --mode test --pred  "head" --db fb15k237

python lstm.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-filtered.json --topk 10 --mode test --pred "head" --db fb15k237

python lstm.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-trainAsTest-topk-10.json  --topk 10 --mode train --pred "tail" --db fb15k237

python lstm.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-unfiltered.json --topk 10 --mode test --pred "tail" --db fb15k237
