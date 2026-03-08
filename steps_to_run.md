# Here are the steps taken to run the scripts:

## Set Up
cd ~/racelab/model_splitting

* import torch:\
conda create -n racelab python=3.11\
conda activate racelab\
pip install torch torchvision pillow

* deactivate .venv prompt\
deactivate

* run script\
conda activate racelab [prompt should be: (racelab) (base)]\
python3 no_splitting_inf.py


## 1. no_splitting_inf.py:
* karen 2/25 results:\
100.0%\
pred idx: 294\
pred label: brown bear\
Throughput: 44.62 images/sec\
Avg latency per image: 22.41 ms


## 2. splitting_inference.py:
* karen 2/25 results:\
pred idx: 294\
pred label: brown bear\
Throughput: 22.86 images/sec


## 3. no_splitting_train.py:
* karen 2/25 results:
step 00  loss=2.3207\
step 05  loss=2.3741\
step 10  loss=2.3136\
step 15  loss=2.3052\
--- Baseline Results (No Splitting) ---\
Steps completed: 16\
Total time: 25.76 s\
Throughput: 9.94 images/sec\
Avg latency per batch: 1610.02 ms


## 4. splitting_training.py:
* karen 2/25 results:\
step 00  loss=2.5625\
step 05  loss=2.2351\
step 10  loss=2.3318\
step 15  loss=2.4282\
Finished 16 steps in 56.39s



#command to run the pipeline splitting file - torch run inference
torchrun --nproc_per_node=2 pipeline_splitting_inf.py \
  --image "bear.jpeg" \
  --stages 2


#on two machines pipeline (godzilla pi version)

#main node (node rank 0) -> aka the node tgat runs the first part of the split model and assigns the second part to the worker node
torchrun --nnodes 2 --nproc-per-node 1 --node-rank 0 --master-addr bramble-4-1-2 --master-port 8123 pipeline_splitting_inf.py --stages 2 --image bear.jpeg


#worker node (node_rank 1)-> aka the node that runs the second part of the split model
torchrun --nnodes 2 --nproc-per-node 1 --node-rank 1 --master-addr bramble-4-1-2 --master-port 8123 pipeline_splitting_inf.py --stages 2 --image bear.jpeg

