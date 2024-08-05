# SPARTUNQChain

This is the code associated with paper https://arxiv.org/abs/2406.13828

## Requirement

Please install the package version in README.md

```python 
pip install -r requirments.txt
```

# Experiment

The possible model option is ["roberta", "t5-adapter", "bert"].

The program will save the parameters of the model in Models folder for any further use.

Data available here: https://drive.google.com/drive/folders/16nBxg1xcPfuQu58Df-PSQZYABsgmk9KQ?usp=sharing.
Note that the augmented Q-Chain part in train_YN_v3.json and train_FR_v3.json on fact_infos parameters


## Yes-No Question

### Baseline
```commandline
python main.py --epoch 8 --train_file ORIGIN --test_file ORIGIN --train_size 1000000 --test_size 1000000 --cuda 0 --lr 8e-6 --batch_size 8
```
### Primal-Dual
```commandline
python main.py --epoch 8 --train_file ORIGIN --test_file ORIGIN --train_size 1000000 --test_size 1000000 --cuda 0 --lr 8e-6 --batch_size 8 --pmd T --constraints T
```
### Primal-Dual + Q-Chain
```commandline
python main.py --epoch 8 --train_file SPARTUN --test_file SPARTUN --train_size 1000000 --test_size 1000000 --cuda 0 --lr 8e-6 --batch_size 8 --model t5-adapter --pmd T --constraints T --save T --save_file Q_chain_T5
```

## Experiment with FR
The possible model option is [ "bert"].
```commandline
python main_rel.py --epoch 8 --train_file ORIGIN --test_file ORIGIN --train_size 1000000 --test_size 1000000 --cuda 0 --lr 8e-6 --batch_size 8
```
### Primal-Dual
```commandline
python main_rel.py --epoch 8 --train_file ORIGIN --test_file ORIGIN --train_size 1000000 --test_size 1000000 --cuda 0 --lr 8e-6 --batch_size 8 --pmd T --constraints T
```
### Primal-Dual + Q-Chain
```commandline
python main_rel.py --epoch 8 --train_file SPARTUN --test_file SPARTUN --train_size 1000000 --test_size 1000000 --cuda 0 --lr 8e-6 --batch_size 8 --pmd T --constraints T --save T --save_file Q_chain_T5
```