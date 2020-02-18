cd openke
bash make.sh
cd ..
#python test_transe_WN18_adv_sigmoidloss.py gpu $1
echo "Find 50 topk answers..."
python train_transe_FB15K237.py gpu 10 "test"
echo "Prepare training for LSTM..."
python "read-answers.py" 10 "/home/uji300/OpenKE/FB15K237-results-scores-10-filtered.json"
echo "Training for LSTM..."
# python program training_file mode topk head/tail predictions
python lstm.py "/var/scratch2/uji300/OpenKE-results/fb15k237-training-topk-10.json" "train" 10 "tail"
echo "Testing for LSTM..."
python lstm.py "/var/scratch2/uji300/OpenKE-results/fb15k237-training-topk-10.json" "train" 10 "tail"
python lstm.py "/var/scratch2/uji300/OpenKE-results/fb15k237-test-topk-10.json" "test" 10 "tail"
