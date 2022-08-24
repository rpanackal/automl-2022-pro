import json
import time
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def plot(path_to_json):
    
    with open(path_to_json, 'r') as json_file:
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


path_to_json = "/home/karma/Documents/AutoML Project/runs/40595.json"
plot(path_to_json)