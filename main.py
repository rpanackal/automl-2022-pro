import numpy as np
import time
import torch
from torch import nn
from dehb import DEHB,ConfigVectorSpace,ConfigSpace
from dataset import OpenMlDataset,Split
from data import MyDataset
from utils import train,test,obj
from torch.utils.data import DataLoader
import rtdl
import argparse
import os
from config import SEED

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = SEED

#Get params
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=40597,  help="Select dataset id")
parser.add_argument("--min_budget", default=20,  help="Select min_budget")
parser.add_argument("--max_budget", default=400,  help="Select min_budget")
parser.add_argument("--min", default=2,  help="Select minutes")
args = parser.parse_args()

#Select dataset
dataset_id =args.dataset
current_dataset = OpenMlDataset(dataset_id,seed=SEED)
current_dataset.train_test_split()

#Run DEHB
rs = np.random.RandomState(seed=SEED)
space = ConfigVectorSpace(
    name="neuralnetwork",
    seed=SEED,
    space={
        "lr": ConfigSpace.UniformFloatHyperparameter("lr", lower=1e-4, upper=1e-3, log=True, default_value=1e-3),
        "dropout_first": ConfigSpace.Float('dropout_first', bounds=(0, 0.9), default=0.34, distribution=ConfigSpace.Normal(mu=0.5, sigma=0.35)),
        "dropout_second": ConfigSpace.Float('dropout_second', bounds=(0, 0.9), default=0.34, distribution=ConfigSpace.Normal(mu=0.5, sigma=0.35)),
        "weight_decay": ConfigSpace.Float("weight_decay", bounds=(0, 2), default=0.1),
        "n_blocks": ConfigSpace.UniformIntegerHyperparameter("n_blocks", lower=4, upper=5, default_value=5),
        "d_main": ConfigSpace.OrdinalHyperparameter("d_main", sequence=[256,512], default_value=512),
        "d_hidden" : ConfigSpace.OrdinalHyperparameter("d_hidden", sequence=[ 256, 512], default_value=512)
    },
)

dehb = DEHB(space, metric='mean_F1',min_budget=int(args.min_budget), max_budget=int(args.max_budget), rs=rs)

start_time = time.process_time()

#Print best configuration
print(f"\n Best configuration  {dehb.optimize(obj, limit=int(args.min),  unit='min', dataset=current_dataset)}")
print(f"Time elapsed (CPU time): {(time.process_time() - start_time):.4f} seconds")
print(f"Validation F1: {dehb.inc_score}")

#save data
dehb.save_data()

#Train best configuration with the whole training set 
train_split = Split(x=current_dataset.train_pred_data,y=current_dataset.train_tar_data)
test_split = Split(x=current_dataset.test_pred_data,y=current_dataset.test_tar_data)
loss_fn = nn.BCEWithLogitsLoss()
n_inputs, n_outputs = current_dataset.train_pred_data.shape[1], current_dataset.train_tar_data.shape[1]
train_dataset = MyDataset(train_split)
test_dataset = MyDataset(test_split)
train_dataloader = DataLoader(train_dataset, batch_size=256)
test_dataloader = DataLoader(test_dataset, batch_size=256)

config = { "d_in":n_inputs,
"n_blocks":dehb.inc_config['n_blocks'] ,
"d_main": dehb.inc_config['d_main'],
"d_hidden":dehb.inc_config['d_hidden'] ,
"dropout_first":dehb.inc_config['dropout_first'],
"dropout_second":dehb.inc_config['dropout_second'],
"d_out":n_outputs}

model = rtdl.ResNet.make_baseline(**config).to(device)  

optimizer = torch.optim.AdamW(model.parameters(), lr= dehb.inc_config['lr'], weight_decay=dehb.inc_config['weight_decay'])
budget = dehb.inc_config['budget']
for t in range(budget):
  train(train_dataloader, model, loss_fn, optimizer)

#Calculate final F1 test score
score_test = test(test_dataloader,model)
print(f"Final Test F1: {score_test}")

#Save best model
torch.save(model.state_dict(), os.path.join(os.path.abspath(os.getcwd()),'model/best.pth'))

