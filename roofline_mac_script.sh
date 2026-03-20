#!/bin/bash

#PRE_REQ activate for pytorch environment if applicable 
log_path="/Users/animeshnd//model_splitting/logs/roofline/"
script_path="/Users/animeshnd//model_splitting/"

for world in {1..5};
do
        # mkdir -p ${log_path}/${world}_size 

        master="127.0.0.1"
        batch_num=1
        # batch_size=4
        iters=2
        for (( batch_size = 1; batch_size < 10; batch_size++ ))
        do
                mkdir -p ${log_path}/${world}_size/${batch_size}/ 

                for (( i = 0 ; i < ${world}; i++ ))
                do
                        val=${good_ones[$i]}
                        python3 ${script_path}/main_infer_exec.py --rank $i --world $world --ip $master --port 8123 --warmup 1 --images /Users/animeshnd//model_splitting/bear.jpeg /Users/animeshnd//model_splitting/penguin.jpeg --batch-size $batch_size --batch-num $batch_num --iters $iters >> ${log_path}/${world}_size/${batch_size}/speed_chronos$i.log &
                        #copy=1 inp_len=1 log_path="${log_path}/trial/" world_size=$world rank=$i master=$master cores=1 srun -N 1 --nodelist=$val ${script_path}/volt_tester_ml.sh 4 0  > /dev/null&
                        #wait
                done
                wait
        done
        # wait
done



