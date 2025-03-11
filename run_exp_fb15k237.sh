#!/bin/sh
source ~/.bashrc

for E in "transe" ; # "complex"
do
    # Generate embeddings -> Horrible performance
    # prun -v -np 1 -t 01:00:00 -native '-C TitanX --gres=gpu:1' ./step1-train-embedding-models.sh "/var/scratch2/uji300/OpenKE-results/" $E "fb15k237"

    # Create subgraphs / clusters
    #prun -v -np 1 -t 01:00:00 -native '-C TitanX --gres=gpu:1' ./step2-create-subgraphs.sh "/var/scratch2/uji300/OpenKE-results/" $E "fb15k237" "star"

    # Test subgraphs
    prun -v -np 1 -t 01:00:00 -native '-C TitanX --gres=gpu:1' ./step3-test-subgraphs.sh -m $E -d "fb15k237" -r -1  -k 10 -s "avg" --type "star"

done

# parameters for test subgraphs
#-m transe -d yago2 -r 100 -k -2 -s kl --type star^C
