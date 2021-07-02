# NegotiationToM
Improving Dialog Systems for Negotiation with Personality Modeling.
Implements our ToM algorithms.

Our code for internal using is available in another [repo](https://github.com/princeton-nlp/cocoa).

A more clear code with proper instruction will be available soon here. 

----------
## Installation

### Dependencies
0. Clone code from github:
```shell
git clone git@github.com:princeton-nlp/NegotiationToM.git
cd NegotiationToM 
```
1. Create a conda environment:
```shell
conda create --name <env> --file tom_conda10.txt
```
2. install pytorch


```shell
# For Linux
# CUDA 9.0
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch

# CUDA 10.0
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

# CPU Only
conda install pytorch-cpu==1.1.0 torchvision-cpu==0.3.0 cpuonly -c pytorch
```

- Check [official website](https://pytorch.org/get-started/previous-versions/) for more installation instructions .
    

3. pip install additional dependencies
```shell
# install other dependecies
pip install -r requirements_cuda10.txt
# install our code
pip install -e .
```

### Dependence Reference
- Python 3.5, pytorch=1.1.0

    - For cuda9: pytorch=1.1.0=py3.5_cuda9.0.176_cudnn7.5.1_0
    - For cuda10: pytorch=1.1.0=py3.5_cuda10.0.130_cudnn7.5.1_0

- Additional dependencies:
    - `tom_new.txt`, `requirements.txt`
    - For cuda10: `tom_cuda10.txt`, `requirements_cuda10.txt`

## Traning instruction

### Preprocess
```shell
PYTHONPATH=. python core/price_tracker.py --train-examples-path data/train-luis-clean2.json --output data/price_tracker.pkl
```
### Train a Supervised Learning Agent
```shell
bash craigslistbargain/exp_scripts/identifier/old/train_sl.sh
```
### Train a Reinforcement Learning Agent
- Generate scenarios
```shell
PYTHONPATH=. python ../scripts/chat_to_scenarios.py --chats data/train-luis-post.json --scenarios data/train-scenarios.json
PYTHONPATH=. python ../scripts/chat_to_scenarios.py --chats data/dev-luis-post.json --scenarios data/dev-scenarios.json
```
- Train the RL model
```shell
bash exp_scripts/rl/train_a2c.sh "_0.001_${i}" "--gpuid 0" "${i}" "0.001" 2>&1 | tee logs/output_train_a2c_0.001_${i}.log &
```
### Train a ToM model
- Sample data
```shell
bash exp_scripts/identifier/sample_data.sh "--gpuid 0" "${i}" 2>&1 | tee logs/sample_data_${i}.log &
```
- Explicit Model
```shell
bash exp_scripts/identifier/train_uttr_history_tom.sh "_${i}" "--gpuid 0" "${i}" 2>&1 | tee logs/uttr_history_tom_${i}.log &
```
- Implicit Model
```shell
bash exp_scripts/identifier/train_uttr_id_history_tom.sh "_${i}" "--gpuid 0" "${i}" 2>&1 | tee logs/uttr_id_history_tom_${i}.log &
```

### Evaluate result of different model.
- Rule Model
```shell
bash exp_scripts/rl/eval_rule.sh "_${i}" "" "${i}" "1" "${CHECK_POINT}" 2>&1 | tee logs/output_eval_rule_${i}.log &
```
- SL Model
```shell
bash exp_scripts/rl/eval_sl.sh "_${i}" "" "${i}" "1" "${CHECK_POINT}" 2>&1 | tee logs/output_eval_sl_${i}.log &
```
- RL Model
```shell
bash exp_scripts/rl/eval_rl.sh "_${i}" "" "${i}" "1" "${CHECK_POINT}" 2>&1 | tee logs/output_eval_rl_${i}.log &
```  
- Explicit ToM Model
```shell
i=0
BETA=5
CHECK_POINT="checkpoint/a2c_0.0001_0/model_reward-0.0865_e1850.pt"
TOM_CHECK_POINT="checkpoint/uttr_history_tom_7_0/model_best.pt"
bash exp_scripts/rl/eval_tom_noid.sh "_${BETA}_${i}" "" "${i}" "${BETA}" "${CHECK_POINT}" "${TOM_CHECK_POINT}"
```
- Implicit ToM Model
```shell
i=0
BETA=20
CHECK_POINT="checkpoint/a2c_0.0001_0/model_reward-0.3006_e1800.pt"
TOM_CHECK_POINT="checkpoint/uttr_id_tom_history_7_0/model_best.pt"
bash exp_scripts/rl/eval_tom.sh "_${BETA}_${i}" "" "${i}" "${BETA}" "${CHECK_POINT}" "${TOM_CHECK_POINT}"
```