import glob
import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


random_forest_baseline = {
    40588 : {
        "dataset": "birds",
        "score": 0.132177
    },
    40589 : {
        "dataset": "emotions",
        "score": 0.624649
    },

    40590 : {
        "dataset": "enron",
        "score": 0.154276
    },

    40591 : {
        "dataset": "genbase",
        "score": 0.716049
    },

    40592 : {
        "dataset": "image",
        "score": 0.471796
    },

    40593 : {
        "dataset": "langLog",
        "score": 0.007548
    },

    40594 : {
        "dataset": "reuters",
        "score": 0.549591
    },

    40595 : {
        "dataset": "scene",
        "score": 0.692282
    },

    40596 : {
        "dataset": "slashdot",
        "score": 0.233592
    },
    40597 : {"dataset": "yeast", "score": 0.326794
    }
}

mlp_scores={ 
    40588: {'dataset': 'birds', 'score': 0.1644815246672832}, 
    40589: {'dataset': 'emotions', 'score': 0.6512038280548539}, 
    40590: {'dataset': 'enron', 'score': 0.17997367613050447}, 
    40591: {'dataset': 'genbase', 'score': 0.7160493827160493}, 
    40592: {'dataset': 'image', 'score': 0.5583341005723621}, 
    40593: {'dataset': 'langLog', 'score': 0.06156451145674692}, 
    40594: {'dataset': 'reuters', 'score': 0.6131403847946456}, 
    40595: {'dataset': 'scene', 'score': 0.7292771021855861}, 
    40596: {'dataset': 'slashdot', 'score': 0.3097921627744437}, 
    40597: {'dataset': 'yeast', 'score': 0.3888284571130446}
    }

def plot(save_path):
    
    with open(save_path, 'r') as json_file:
        d = json.load(json_file)
        json_file.close()
     
    del d["history"]

    runtimes= d["runtime"]
    traj = d["traj"]
    # Creating the Figure instance

    df = pd.DataFrame({"Wallclock time": runtimes, "Incubent F1-score": traj})
    trace0 = sns.lineplot(x='Wallclock time', y='Incubent F1-score', data=df)
    trace0.set(xscale='log', yscale="log")

    plt.title('F1_score vs Wallclock time')
    sns.set_context("paper")
    plt.show()

def best_config_score_list(save_dir):

    dd = defaultdict(list)
    index = []
    for file in glob.glob(f"{save_dir}/*.json"):
        if "_" in file : continue
        dataset_id = Path(file).stem
        with open(file, "r") as json_file:
            d = json.load(json_file)
            json_file.close()
        
        index.append(int(dataset_id))
        dd["dataset"].append(random_forest_baseline[int(dataset_id)]["dataset"])
        dd["RF/score"].append(random_forest_baseline[int(dataset_id)]["score"])
        dd["MLP/score"].append(mlp_scores[int(dataset_id)]["score"])
        dd["DEHB/score"].append(d["result"]["best_score"])
        
        for key, value in d["result"]["best_config"].items():
            dd[key].append(value)
    
    df = pd.DataFrame(dd, index=index)
    #df.set_index("id")
    df = df.sort_index()
    print(df)




save_dir = "/home/karma/Documents/AutoML Project/runs"
save_path = save_dir + "/40596.json"
plot(save_path)
best_config_score_list(save_dir)
