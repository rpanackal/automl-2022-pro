# End to end deep multilabel classification

### Description
------------
End to end system for multilabel classification using ResNet blocks and Differential Evolution Hyperband

### Requirements
------------
In a conda or virtual env, install the requirements.

`pip install -r requirements.txt`

### Usage (how to reproduce results)
------------

To run the ResNet with DEHB and run:
```bash
$ python main.py --dataset 40589 --min_budget 20 --max_budget 400 --min 55
>Validation F1: 0.634457385678587
>Final Test F1: 0.650024360748846
```
List of command line arguments
```bash
--dataset     dataset id
--min_budget  minimum number of epochs for DEHB
--max_budget  maximum number of epochs for DEHB
--min         number of minutes DEHB has to get the best configuration 
```


List of available datasets
```bash
name        id
birds       40588
emotions    40589
enron       40590
genbase     40591
image       40592
langLog     40593
reuters     40594
scene       40595
slashdot    40596
yeast       40597
```