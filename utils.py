import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from metrics import f1_score
import numpy as np
import torch
from torch import nn
from dataset import  CrossValidation,Split
from data import MyDataset
from torch.utils.data import DataLoader
import rtdl
import wandb

SEED = 42

# Get cpu or gpu device for training.
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
from torch.functional import split



def train(dataloader, model, loss_fn, optimizer):
    """Train function

    Parameters
    ----------
    dataloader : Dataloader
        Pytorch dataloader

    model: torch.nn.Module
        Pytorch model

    loss_fn: nn.BCEWithLogitsLoss
        Binary cross-entropy loss

    optimizer: torch.optim.AdamW
        Adam with decoupled weight decay

    Returns
    -------
    none
    """
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X) 
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       

def test(dataloader, model):
    """Train function

    Parameters
    ----------
    dataloader : Dataloader
        Pytorch dataloader

    model: torch.nn.Module
        Pytorch model

    Returns
    -------
    int
        F1 score
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    #print("num batches: "+str(num_batches))
    #print("size: "+str(size))
    model.eval()
    f1 = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X) 

            pred = torch.sigmoid(pred)

            yhat = pred.round()
            # calculate accuracy
            f1 += f1_score(y.detach().cpu().numpy(), yhat.detach().cpu().numpy())
    f1 /= num_batches
    return f1
    
def obj(x,budget,**kwargs):
    """Objective function used by DEHB

    Parameters
    ----------
    x : ConfigVectorSpace
        Dictionary with configuration

    budget : int | None = None
        number of epochs the training will run

    Returns
    -------
    dict{}
        average f1 score and std
    """
    seed = SEED
    dataset = kwargs['dataset']
    
    cross_validation = CrossValidation(pred_data=dataset.train_pred_data, target_data=dataset.train_tar_data, split_num=3)

    score_tests = np.array([])

    for i, (tp, tt, vp, vt) in enumerate(cross_validation):
        loss_fn = nn.BCEWithLogitsLoss()


        train_split = Split(x=tp,y=tt)
        validation_split = Split(x=vp,y=vt)

        
        train_dataset = MyDataset(train_split)
        test_dataset = MyDataset(validation_split)

        
        train_dataloader = DataLoader(train_dataset, batch_size=256)
        test_dataloader = DataLoader(test_dataset, batch_size=256)

        n_inputs, n_outputs = tp.shape[1], tt.shape[1]

        config = { "d_in":n_inputs,
        "n_blocks":x["n_blocks"],
        "d_main":x["d_main"],
        "d_hidden":x["d_hidden"],
        "dropout_first":x["dropout_first"],
        "dropout_second":x["dropout_second"],
        "d_out":n_outputs}

        model = rtdl.ResNet.make_baseline(**config).to(device)  

        config["batch_size"] = 256
        config["epoch"] = budget
        config["weight_decay"] = x['weight_decay']

        optimizer = torch.optim.AdamW(model.parameters(), lr=x["lr"], weight_decay=x['weight_decay'])
        #Only save the first run
        if i==0:
          with wandb.init(
                  project="multilabel",
                  config=config,
                  group=str(dataset.dataset_id),
                  reinit=True,
                  mode="online",
                  settings=wandb.Settings(start_method="thread")):

              for t in range(budget):
                  train(train_dataloader, model, loss_fn, optimizer)
                  score_test = test(test_dataloader,model)
                  wandb.log({"val/f1_score": score_test})

              score_tests = np.append(score_tests,score_test)
        else:
            for t in range(budget):
                  train(train_dataloader, model, loss_fn, optimizer)
                  score_test = test(test_dataloader,model)

            score_tests = np.append(score_tests,score_test)

    print(config)
    print("Mean f1 score of configuration: " + str(np.mean(score_tests)))
    return {'mean_F1':np.mean(score_tests), 'std_F1':np.std(score_tests)}