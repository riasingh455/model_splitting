#!/bin/bash 

model_type=$1
model_split=$2
node_prefix=$3
cores=$4
path_prefix="/home/animesh/test_model_split/"

if [[ -z $model_type ]] || [[ -z $model_split ]] || [[ -z $node_prefix ]] || [[ -z $cores ]]
then
	echo "Missing arguments, expected: ./script.sh <model_type> (resnet18, mbv3_small, eb0) <model_split> (children, modules) <node_prefix> (bramble-x-y) <number of cores> (int, max 4)" && exit
fi

all_nodes=$(sinfo -N | grep "idle" | grep -v "idle\*" | grep "${node_prefix}" | awk '{print$1}')

valid_pool_dir_path="/home/animesh/test_model_split/aot_splitter/${node_prefix}/"
mkdir -p valid_pool_dir_path

# for repeat in {1..10}
for repeat in 0
do  
    all_devs=()
    echo -e "\n run ${repeat} start \n" >> ${valid_pool_dir_path}/valid_mem.pool 
    path_dst="${path_prefix}/logs/single_group_heat_con/${node_prefix}/per_dev/${repeat}/"
    mkdir -p ${path_dst}
    for (( n=0;n<${#all_nodes[@]};n++ ))
    do
        mem=$(timeout 1m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@$n_check "pkill -9 -f temp_speed_reader.sh; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';free | grep 'Mem'" | awk '{print $4}')
        if [[ $mem -gt 450000 ]]
        then
            all_valid_devs+=(${all_nodes[$n]})
            echo "${all_nodes[$n]}" >> ${valid_pool_dir_path}/valid_mem.pool 
            command="python3 onnxtest.py 5 ./vit_modules_3_1_custom 1 4 $(date -d '+1 seconds' +%s) > ${path_dst}/speed_chronos${all_nodes[$n]}.log"
            full_command="pushd ${path_prefix}/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate;${command} &"
            hcommand="mac_ver=0 log_path=${path_dst} ${script_path}/temp_speed_reader.sh"
            full_hcommand="pushd ${path_prefix}/aot_splitter;${hcommand} &"
            
            timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${all_nodes[$n]} "${full_command}" &
            pause_pid=$!
            while [[ $(cat ${path_dst}/speed_chronos${all_nodes[$n]}.log | grep "Sync done -> model run start" | wc -l) -lt 1 ]]
            do
                continue
            done
            ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${all_nodes[$n]} "${full_hcommand}" &
            kill_pid=$!
            wait $pause_pid
            kill -9 $kill_pid
        fi
    done
    
    for n in ${all_valid_devs[@]}
    do
            mem=$(timeout 1m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@$n_check "pkill -9 -f temp_speed_reader.sh; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';free | grep 'Mem'" | awk '{print $4}')
    done

    path_dst="${path_prefix}/logs/single_group_heat_con/${node_prefix}/all_dev/${repeat}/"
    mkdir -p ${path_dst}
    waiters=()
    killers=()
    dt=$(date -d '+120 seconds' +%s)
        for n in ${all_valid_devs[@]}
        do
                command="python3 onnxtest.py 5 ./vit_modules_3_1_custom 1 4 ${dt} > ${path_dst}/speed_chronos${n}.log"
                full_command="pushd ${path_prefix}/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate;${command} &"
                hcommand="mac_ver=0 log_path=${path_dst} ${script_path}/temp_speed_reader.sh"
                full_hcommand="pushd ${path_prefix}/aot_splitter;${hcommand} &"
                
                timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${n} "${full_command}" &
                waiters+=($!)

                ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${n} "sleep 60; ${full_hcommand}" &
                killers+=($!)
        done
    wait ${waiters[@]}
    kill -9 ${killers[@]}
done
