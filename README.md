# Deep Learning course project on Active Learning
Deep neural models for active learning in the sequence tagging tasks

## Repository structure
- `data` -- a place with training and evaluation datasets on which we conducted our experiments
- `experiments` -- scores, that we got after experimenting with different models and strategies. Each result folder consists
of several runs in order to provide the variance estimation
- `src` -- main scripts to run experiments on your own
- `configs` -- main files with paramethers to run the experiments

## Running example

Suppose you want to run an Active Learning emulation using BERT with Monte-Carlo Dropout extension

To run the experiment in our setting you should write the config pass and the name of our main script, as following:
    
    HYDRA_CONFIG_PATH='../configs/al_exp_bert.yaml' python run_active_learning.py

Also, you can determine the desired cuda device (not the 0 one) before the config path:

    CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH='../configs/al_exp_bert.yaml' python run_active_learning.py

You can enable multiprocessing via run_al_multiprocess.py. It works in the same way as run_active_learning.py. It will consume more RAM, please use it on your own risk!
    
Scripts assume that you do not already have statistics for the experiment in the `experiments` directory.

*Keep in mind, that computations on large datasets like CoNLL-2003 can be rather heavy and assume that you are running program on gpu and have 6--8 GB of VRAM to spare.*

PS We ran the experiments using clasters with several Nvidia GeForce 2080 and Tesla P40. 10GB of VRAM should be more than enough for a single run for the most computationaly expensive part.

---
Collaborators: Denis Belyakov, Lyubov Kuprianova, Miron Kuznetsov, Dmitri Puzyrev, Angelina Yaroshenko

*Special thanks to our research supervisor Artem Shelmanov!*
