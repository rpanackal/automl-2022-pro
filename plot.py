from matplotlib import pyplot as plt
from pathlib import Path
import json
import glob

id_to_name={ 
    40588: 'birds',
    40589: 'emotions',
    40590: 'enron',
    40591: 'genbase',
    40592: 'image',
    40593: 'langLog',
    40594: 'reuters',
    40595: 'scene',
    40596: 'slashdot',
    40597: 'yeast',
    }

def plot(save_dir=None):

    fig, ax = plt.subplots(2,5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    colors = ["red", "black", "darkorchid", "blue", "darkgreen", "brown", "orange", "cyan", "magenta", "lightgreen"]

    for i, file in enumerate(glob.glob(f"{save_dir}/*.json")):
        
        dataset_id = Path(file).stem
        #if int(dataset_id) != 40590 : continue
        dataset_name = id_to_name[int(dataset_id)]

        with open(file, 'r') as json_file:
            d = json.load(json_file)
            json_file.close()
     
        del d["history"]

        runtimes= d["runtime"]
        traj = d["traj"]

        # # Creating the Figure instance
        row = 0 if i <5 else 1
        col = i % 5
        
        ax[row][col].plot(runtimes, traj, color=colors[i])#, label=dataset_name)
        ax[row][col].set_xlabel('time (sec)')
        ax[row][col].set_ylabel('f1-score')
        ax[row][col].set_title(dataset_name)
        ax[row][col].loglog()

    fig.suptitle('Val/F1_score vs Wallclock time', fontsize=12)
    plt.show()

save_dir = "./logs"
plot(save_dir=save_dir)