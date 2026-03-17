#!/bin/bash

source /home/animesh/model_splitting/pi-torch/bin/activate 

temp_log_path="/home/animesh/model_splitting/logs"

if [ -z $log_path ]
then
	log_path=$temp_log_path
fi

script_path="/home/animesh/model_splitting/"
t=$(hostname)
log_path=${log_path} ${script_path}/temp_speed_reader.sh &
last_pid=$!
pids=()
k=0
#for (( k=$start; k<=$end; k++ )); do
echo "startheat ${k}-$(date '+TIME:%H:%M:%S.%3N')" >> ${log_path}/speed_chronos$t.log

echo "-${k}- number of processes running"
#for (( j=1; j<=$k; j++ )); do
     #run buniel on cores
#torchrun --nnodes $world_size --nproc-per-node 1 --node-rank $rank --master-addr $master --master-port 8123 ${script_path}/main_infer_exec.py --warmup 3 --cores $cores  >> ${log_path}/speed_chronos$t.log  &
python3 main_infer_exec.py --rank $rank --world $world_size --ip $master --port 8123 --warmup 1 --images /home/animesh/model_splitting/bear.jpeg /home/animesh/model_splitting/penguin.jpeg --batch-size $batch_size --batch-num $batch_num --iters $iters >> ${log_path}/speed_chronos$t.log &
#repeat images 30 times

#img1="/home/pi/model_splitting/bear.jpeg"
#img2="/home/pi/model_splitting/penguin.jpeg"
#final_str=""
#for (( i = 0; i < $inp_len; i++ ))
#do
#	final_str+="${img1} ${img2} "
#done

#j=0
#for (( j = 0; j < $copy; j++ ))
#do
#	torchrun --nnodes $world_size --nproc-per-node 1 --node-rank $rank --master-addr $master --master-port $(( 8123+$j )) ${script_path}/main_infer_exec.py --warmup 0 --iters 1 --cores $cores --images $final_str >> ${log_path}/speed_chronos$t.log  &
pids+=($!)
#done

      # echo "${i} ${j} loop"
  #done
wait "${pids[@]}"

echo "endheat ${k}-$(date '+TIME:%H:%M:%S.%3N')" >> ${log_path}/speed_chronos$t.log
 # echo "startcool ${k}-$(date '+TIME:%H:%M:%S.%3N')" >> ${log_path}/${size}_logs/speed_chronos$t.log
 # sleep 60
 # echo "endcool ${k}-$(date '+TIME:%H:%M:%S.%3N')" >> ${log_path}/${size}_logs/speed_chronos$t.log
#done

kill -9 $last_pid
