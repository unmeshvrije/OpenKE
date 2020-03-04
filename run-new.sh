#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-trainAsTest-topk-10.json  --topk 10 --mode train --pred head --db fb15k237 --units 500 --dropout 0.5

#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-filtered.json  --topk 10 --mode "test" --pred head --db fb15k237 --units 500 --dropout 0.5

#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-unfiltered.json  --topk 10 --mode "test" --pred head --db fb15k237 --units 500 --dropout 0.5

#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-trainAsTest-topk-10.json  --topk 10 --mode train --pred head --db fb15k237 --units 500 --dropout 0.8

#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-filtered.json  --topk 10 --mode "test" --pred head --db fb15k237 --units 500 --dropout 0.8

#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-unfiltered.json  --topk 10 --mode "test" --pred head --db fb15k237 --units 500 --dropout 0.8

#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-trainAsTest-topk-10.json  --topk 10 --mode train --pred head --db fb15k237 --units 50 --dropout 0.5

#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-filtered.json  --topk 10 --mode "test" --pred head --db fb15k237 --units 50 --dropout 0.5

#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-unfiltered.json  --topk 10 --mode "test" --pred head --db fb15k237 --units 50 --dropout 0.5

#############################
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-trainAsTest-topk-10.json  --topk 10 --mode train --pred "tail" --db fb15k237 --units 50 --dropout 0.2 --model "lstm"

#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-filtered.json  --topk 10 --mode "test" --pred "tail" --db fb15k237 --units 50 --dropout 0.2 --model "lstm"

#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-unfiltered.json  --topk 10 --mode "test" --pred "tail" --db fb15k237 --units 50 --dropout 0.2 --model "lstm"

#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-trainAsTest-topk-10.json  --topk 10 --mode train --pred "head" --db fb15k237 --units 50 --dropout 0.2 --model "lstm"

#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-filtered.json  --topk 10 --mode "test" --pred "head" --db fb15k237 --units 50 --dropout 0.2 --model "lstm"

#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-unfiltered.json  --topk 10 --mode "test" --pred "head" --db fb15k237 --units 50 --dropout 0.2 --model "lstm"


#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-trainAsTest-topk-10.json  --topk 10 --mode train --pred "tail" --db fb15k237 --units 50 --dropout 0.2 --model "mlp"


#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-filtered.json  --topk 10 --mode "test" --pred "tail" --db fb15k237 --units 50 --dropout 0.2 --model "mlp"

#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-unfiltered.json  --topk 10 --mode "test" --pred "tail" --db fb15k237 --units 50 --dropout 0.2 --model "mlp"

#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-trainAsTest-topk-10.json  --topk 10 --mode train --pred "head" --db fb15k237 --units 50 --dropout 0.2 --model "mlp"

#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-filtered.json  --topk 10 --mode "test" --pred "head" --db fb15k237 --units 50 --dropout 0.2 --model "mlp"

#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-unfiltered.json  --topk 10 --mode "test" --pred "head" --db fb15k237 --units 50 --dropout 0.2 --model "mlp"


# head tests ****************************
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-trainAsTest-topk-10.json  --topk 10 --mode train --pred "head" --db fb15k237 --units 5 --dropout 0.2 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-filtered.json  --topk 10 --mode "test" --pred "head" --db fb15k237 --units 5 --dropout 0.2 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-unfiltered.json  --topk 10 --mode "test" --pred "head" --db fb15k237 --units 5 --dropout 0.2 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-trainAsTest-topk-10.json  --topk 10 --mode train --pred "head" --db fb15k237 --units 500 --dropout 0.1 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-filtered.json  --topk 10 --mode "test" --pred "head" --db fb15k237 --units 500 --dropout 0.1 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-unfiltered.json  --topk 10 --mode "test" --pred "head" --db fb15k237 --units 500 --dropout 0.1 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-trainAsTest-topk-10.json  --topk 10 --mode train --pred "head" --db fb15k237 --units 500 --dropout 0.0 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-filtered.json  --topk 10 --mode "test" --pred "head" --db fb15k237 --units 500 --dropout 0.0 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-unfiltered.json  --topk 10 --mode "test" --pred "head" --db fb15k237 --units 500 --dropout 0.0 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-trainAsTest-topk-10.json  --topk 10 --mode train --pred "head" --db fb15k237 --units 500 --dropout 0.5 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-filtered.json  --topk 10 --mode "test" --pred "head" --db fb15k237 --units 500 --dropout 0.5 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-unfiltered.json  --topk 10 --mode "test" --pred "head" --db fb15k237 --units 500 --dropout 0.5 --model "mlp"

# tail tests ********
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-trainAsTest-topk-10.json  --topk 10 --mode train --pred "tail" --db fb15k237 --units 5 --dropout 0.2 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-filtered.json  --topk 10 --mode "test" --pred "tail" --db fb15k237 --units 5 --dropout 0.2 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-unfiltered.json  --topk 10 --mode "test" --pred "tail" --db fb15k237 --units 5 --dropout 0.2 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-trainAsTest-topk-10.json  --topk 10 --mode train --pred "tail" --db fb15k237 --units 500 --dropout 0.1 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-filtered.json  --topk 10 --mode "test" --pred "tail" --db fb15k237 --units 500 --dropout 0.1 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-unfiltered.json  --topk 10 --mode "test" --pred "tail" --db fb15k237 --units 500 --dropout 0.1 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-trainAsTest-topk-10.json  --topk 10 --mode train --pred "tail" --db fb15k237 --units 500 --dropout 0.0 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-filtered.json  --topk 10 --mode "test" --pred "tail" --db fb15k237 --units 500 --dropout 0.0 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-unfiltered.json  --topk 10 --mode "test" --pred "tail" --db fb15k237 --units 500 --dropout 0.0 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-trainAsTest-topk-10.json  --topk 10 --mode train --pred "tail" --db fb15k237 --units 500 --dropout 0.5 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-filtered.json  --topk 10 --mode "test" --pred "tail" --db fb15k237 --units 500 --dropout 0.5 --model "mlp"
#
#python answer_model.py --infile /var/scratch2/uji300/OpenKE-results/fb15k237/lstm-fb15k237-transe-results-scores-test-topk-10-unfiltered.json  --topk 10 --mode "test" --pred "tail" --db fb15k237 --units 500 --dropout 0.5 --model "mlp"
