#prun -v -np 1 -t 01:00:00 -native '-C gpunode --gres=gpu:1' ./supervised.sh transe fb15k237 tail
#prun -v -np 1 -t 01:00:00 -native '-C gpunode --gres=gpu:1' ./supervised.sh transe fb15k237 head
prun -v -np 1 -t 01:00:00 -native '-C gpunode --gres=gpu:1' ./supervised.sh complex fb15k237 tail
prun -v -np 1 -t 01:00:00 -native '-C gpunode --gres=gpu:1' ./supervised.sh complex fb15k237 head
prun -v -np 1 -t 01:00:00 -native '-C gpunode --gres=gpu:1' ./supervised.sh rotate fb15k237 tail
prun -v -np 1 -t 01:00:00 -native '-C gpunode --gres=gpu:1' ./supervised.sh rotate fb15k237 head
