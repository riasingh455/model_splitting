#!/bin/bash

rng_gen () {
        len=$1
        rng_arr=()
        while [[ ${#rng_arr[@]} -lt $conc_val ]]
        do
                rng_arr+=($[ $RANDOM % $len ])
                rng_arr=( $(for i in ${rng_arr[@]}; do echo $i; done | sort -u) )
        done
        echo "${rng_arr[@]}"
}

node_select () {
        # valid_nodes=($(cat ${path_prefix}/full_sweep/${node_prefix}_middling))
        all_nodes=$(sinfo -N | grep "idle" | grep -v "idle\*" | grep "${node_prefix}" | awk '{print$1}')
        #node_list=( $(for i in ${all_nodes[@]}; do [[ $(cat ${path_prefix}/full_sweep/${node_prefix}_middling | grep -w "${i}" | wc -l) -gt 0 ]] && echo "${i}"; done) )
	node_list=()
        for n_check in ${all_nodes[@]}; 
	do
		# if [[ $(cat ${path_prefix}/full_sweep/${node_prefix}_middling | grep -w "${n_check}" | wc -l) -gt 0 ]]
		if [[ $(cat ${path_prefix}/aot_splitter/${node_prefix}.model.perf | grep -w "${n_check}" | wc -l) -gt 0 ]]
		then
			hname=$(ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@$n_check 'hostname')
			mem=$(ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@$n_check "echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';free | grep 'Mem'" | awk '{print $4}')
			if [[ $n_check == $hname ]]
			then
				node_list+=($n_check)
			fi
		fi
	done
	#echo "list ${node_list[@]}"

        ind_arr=( $(rng_gen "${#node_list[@]}") )
        node_select=( $(for i in ${ind_arr[@]}; do echo "${node_list[$i]}"; done) )
        echo "${node_select[@]}"
}

p_2 () {
    #for 0 
    counter=0
    path_dst="${path_prefix}/logs/full_pipeline/${node_prefix}/${model_type}_${model_split}_onnx/${counter}/${repeat}/"
    mkdir -p ${path_dst}
    counter=$(( $counter + 1 ))
    timeout 2m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pkill -9 temp_speed; pkill -9 pipeline; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
    dt=$(date -d '+1 seconds' +%s)   
    command="python3 onnxtest.py 10 ./${model_type}_${model_split}_${fake_world}_1_custom 0 4 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${nodes[0]}_${fake_world}_0.log &" &
    wait $!

    #for 0,1
    waiters=()
    path_dst="${path_prefix}/logs/full_pipeline/${node_prefix}/${model_type}_${model_split}_onnx/${counter}/${repeat}/"
    mkdir -p ${path_dst}
    counter=$(( $counter + 1 ))
    timeout 2m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pkill -9 temp_speed; pkill -9 pipeline; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
    timeout 2m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pkill -9 temp_speed; pkill -9 pipeline; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
    dt=$(date -d '+60 seconds' +%s)   
    command="python3 onnxtest.py 10 ./${model_type}_${model_split}_${fake_world}_1_custom 0 4 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[0]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${nodes[0]}_${fake_world}_0.log &" &
    waiters+=($!)
    command="python3 onnxtest.py 10 ./${model_type}_${model_split}_${fake_world}_1_custom 1 4 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${nodes[1]}_${fake_world}_1.log &" &
    waiters+=($!)

    wait ${waiters[@]}

    #for 1
    waiters=()
    path_dst="${path_prefix}/logs/full_pipeline/${node_prefix}/${model_type}_${model_split}_onnx/${counter}/${repeat}/"
    mkdir -p ${path_dst}
    counter=$(( $counter + 1 ))
    timeout 2m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pkill -9 temp_speed; pkill -9 pipeline; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
    dt=$(date -d '+1 seconds' +%s)   
    command="python3 onnxtest.py 10 ./${model_type}_${model_split}_${fake_world}_1_custom 1 4 ${dt}"
    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[1]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${nodes[1]}_${fake_world}_1.log &" &
    wait $!

}

p_5 () {
    counter=0
    for end in 1 2 3 4 5
    do
        waiters=()
        dt=$(date -d '+60 seconds' +%s)   
        for (( start=0;start<$end;start++ ))
        do
            path_dst="${path_prefix}/logs/full_pipeline/${node_prefix}/${model_type}_${model_split}_onnx/${counter}/${repeat}/"
            mkdir -p ${path_dst}
            counter=$(( $counter + 1 ))
            timeout 30s ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$start]} "pkill -9 temp_speed; pkill -9 pipeline; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
            command="python3 onnxtest.py 10 ./${model_type}_${model_split}_${fake_world}_1_custom ${start} 4 ${dt}"
            timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$start]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${nodes[$start]}_${fake_world}_${start}.log &" &
            waiters+=($!)
        done
        wait ${waiters[@]}
    done

    for end in 1 2 3 4
    do
        waiters=()
        dt=$(date -d '+60 seconds' +%s)   
        for (( start=$end;start<5;start++ ))
        do
            path_dst="${path_prefix}/logs/full_pipeline/${node_prefix}/${model_type}_${model_split}_onnx/${counter}/${repeat}/"
            mkdir -p ${path_dst}
            counter=$(( $counter + 1 ))
            timeout 30s ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$start]} "pkill -9 temp_speed; pkill -9 pipeline; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
            command="python3 onnxtest.py 10 ./${model_type}_${model_split}_${fake_world}_1_custom ${start} 4 ${dt}"
            timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$start]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${nodes[$start]}_${fake_world}_${start}.log &" &
            waiters+=($!)
        done
        wait ${waiters[@]}
    done
}



model_type=$1
model_split=$2
node_prefix=$3
# cores=$4
path_prefix="/home/animesh/test_model_split/"

# if [[ -z $model_type ]] || [[ -z $model_split ]] || [[ -z $node_prefix ]] || [[ -z $cores ]]
if [[ -z $model_type ]] || [[ -z $model_split ]] || [[ -z $node_prefix ]] 
then
	# echo "Missing arguments, expected: ./script.sh <model_type> (resnet18, mbv3_small, eb0) <model_split> (children, modules) <node_prefix> (bramble-x-y) <number of cores> (int, max 4)" && exit
	echo "Missing arguments, expected: ./script.sh <model_type> (resnet18, mbv3_small, eb0) <model_split> (children, modules) <node_prefix> (bramble-x-y)" && exit
fi

# p_2=( 0 (0 1) 1 )
# p_5=( 0 (0 1) (0 1 2) (0 1 2 3) (0 1 2 3 4) (1 2 3 4) (2 3 4) 4 )
# for repeat in {1..10}
for repeat in 0
do
    for fake_world in 5 2
    # for fake_world in 5
    do
        conc_val=$fake_world
    	nodes=( $(node_select) )
        if [[ $fake_world -eq 5 ]]
        then
            p_5
        else
            p_2
        fi
    done
done



