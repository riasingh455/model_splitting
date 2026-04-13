#!/bin/bash

#PRE_REQ activate for pytorch environment if applicable 
# log_path="/home/animesh//model_splitting/logs/roofline/"
# log_path="/home/animesh//model_splitting/logs/roofline/resnet_18"
# log_path="/home/animesh//model_splitting/logs/roofline/ef_b0"

killer () {
        kill -9 $(ps -aex | grep "main_infer_exec.py" | awk '{print $1}')
        #kill -9 $(ps -aex | grep "temp_speed_reader.sh" | awk '{print $1}')
        kill -9 $(ps -aex | grep "roofline" | awk '{print $1}')
}

timeout () {
        echo "timeout started for ${log_path}/${world}_size/${batch_size}_${batch_num}/speed_chronos${t}.log"
        sleep 600
        keyword=$(cat "${log_path}/${world}_size/${batch_size}_${batch_num}/speed_chronos${t}.log" | grep "Sync done -> model run start" | wc -l)
        if [[ $keyword -ge 1 ]]
        then
                echo "checking second timeout"
                sleep 600
                if [[ $( cat ${log_path}/${world}_size/${batch_size}_${batch_num}/speed_chronos${t}.log | grep "rank"  | wc -l ) -eq 0 ]]
                then 
                        echo "died by second timeout"
                        killer
                else
                        echo "timeout for ${log_path}/${world}_size/${batch_size}_${batch_num}/speed_chronos${t}.log not triggered!"
                        return
                fi

        else
                echo "died by timeout"
                killer
        fi
        
}


source /home/animesh/model_splitting/pi-torch/bin/activate
log_path="/home/animesh/test_model_split/logs/roofline/"
script_path="/home/animesh/test_model_split/aot_splitter/"

t=$(hostname)

if [[ -z $model_type ]] || [[ -z $model_split ]]
then
        echo "Please enter model type (resnet18, mbv3_small, eb0) and model split (children, modules)" && exit
fi

log_path="/home/animesh/test_model_split/logs/roofline/${node_prefix}/${model_type}_${model_split}/"


iters=30
pushd ${script_path}

for batch_num in 1 2 5 10
do
        # for batch_size in 2 4 6 8 
        for batch_size in 1
        # for batch_size in 2
        do
                pids=()
                mkdir -p ${log_path}/${world}_size/${batch_size}_${batch_num}/ 
                
                # val=${good_ones[$i]}
                # python3 ${script_path}/main_infer_exec.py --cores 4 --rank $rank --world $world --ip $master --port 8123 --warmup 1 --images /home/animesh//model_splitting/bear.jpeg /home/animesh//model_splitting/penguin.jpeg --batch-size $batch_size --batch-num $batch_num --iters $iters --model-type $model_type --model-split-type $model_split >> ${log_path}/${world}_size/${batch_size}_${batch_num}/speed_chronos${t}.log &
                python3 ${script_path}/main_runner.py --cores 4 --rank $rank --world $world --ip $master --port 8123 --warmup 1 --batch-size $batch_size --batch-num $batch_num --iters $iters --image ${script_path}/bear.jpeg --model-type $model_type --model-split-type $model_split >> ${log_path}/${world}_size/${batch_size}_${batch_num}/speed_chronos${t}.log &
                pids+=($!)
                #copy=1 inp_len=1 log_path="${log_path}/trial/" world_size=$world rank=$i master=$master cores=1 srun -N 1 --nodelist=$val ${script_path}/volt_tester_ml.sh 4 0  > /dev/null&
                #wait

                timeout &

                while true; 
                do      
                        keyword=$(cat "${log_path}/${world}_size/${batch_size}_${batch_num}/speed_chronos${t}.log" | grep "Sync done -> model run start" | wc -l)
                        if [[ $keyword -ge 1 ]]
                        then
                                break
                        fi
                        sleep 1
                done
                #forced to use 0 since all of this on the same machine
                #mac_ver=0 log_path="${log_path}/${world}_size/${batch_size}_${batch_num}/" ${script_path}/temp_speed_reader.sh &
                #last_pid=$!


                wait "${pids[@]}"


                #if logs failed, end exp then and there
                #0 as a file should always be there, if not, even more reasons to kill the experiment right here and now
                if [[ $( cat ${log_path}/${world}_size/${batch_size}_${batch_num}/speed_chronos${t}.log | grep "rank"  | wc -l ) -eq 0 ]]
                then
                        #kill -9 $last_pid
                        kill -9 "${pids[@]}"
                        exit
                fi
                #kill -9 $last_pid
        done
        # wait
done

popd

