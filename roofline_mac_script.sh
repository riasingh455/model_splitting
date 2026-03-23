#!/bin/bash

#PRE_REQ activate for pytorch environment if applicable 
# log_path="/Users/animeshnd//model_splitting/logs/roofline/"
# log_path="/Users/animeshnd//model_splitting/logs/roofline/resnet_18"
# log_path="/Users/animeshnd//model_splitting/logs/roofline/ef_b0"
log_path="/Users/animeshnd//model_splitting/logs/roofline/mb_small"
script_path="/Users/animeshnd//model_splitting/"

for world in {1..100};
# for world in 1 5 10 20 50 100;
# for world in 10;
do
        # mkdir -p ${log_path}/${world}_size 

        master="127.0.0.1"
        batch_num=1
        # batch_size=4
        iters=2
        #(( batch_size = 1; batch_size < 10; batch_size++ ))
        for batch_size in 2 4 6 8 
        # for batch_size in 2
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

                #if logs failed, end exp then and there
                #0 as a file should always be there, if not, even more reasons to kill the experiment right here and now
                if [[ $( cat ${log_path}/${world}_size/${batch_size}/speed_chronos0.log | grep "Time taken by rank"  | wc -l ) -eq 0 ]]
                then
                        exit
                fi
        done
        # wait
done



