import openke
from openke.utils import DeepDict
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import json

import sys
# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/WN18RR/", 
	batch_size = 2000,
	threads = 8,
	sampling_mode = "cross", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 64,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 1024, 
	p_norm = 1,
	norm_flag = False,
	margin = 6.0)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = SigmoidLoss(adv_temperature = 1),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.0
)

is_gpu = sys.argv[1] == 'gpu'
# train the model
#trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 30, alpha = 2e-5, use_gpu = is_gpu, opt_method = "adam")
#trainer.run()
#transe.save_checkpoint('./checkpoint/transe_2.ckpt')

# test the model
tester = Tester("WN18RR", model = transe, data_loader = test_dataloader, use_gpu = is_gpu)
tester.run_link_prediction(type_constrain = False)
transe.load_parameters('./result/WN18-transe.json')
