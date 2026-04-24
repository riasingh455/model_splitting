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

#for repeat in 0
for repeat in {1..10}
do 
    for world in 2 5
    do 
        conc_val=$world
        nodes=( $(node_select) )
        nodes_string=''
        for i in ${nodes[@]}; do nodes_string+="${i},";done;
        # 1 batch size/iters, 10 batch num
        #path_dst="${path_prefix}/logs/single_group_heat_con/${node_prefix}/${model_type}_${model_split}_zmq_bs_1_bn_10/${repeat}/"
        path_dst="${path_prefix}/logs/single_group_heat_con/${node_prefix}/${model_type}_${model_split}_zmq_bs_5_bn_5/${repeat}/"
	mkdir -p ${path_dst}
        waiters=()
        killers=()
        for (( n=0;n<${#nodes[@]};n++ ))
        do
		#command="python3 onnx_w_zmq.py 1,5 ./${model_type}_${model_split}_${world}_1_custom ${n},${world} 4 $(( 9123+$world )) ${nodes_string}"
		command="python3 onnx_w_zmq.py 5,5 ./${model_type}_${model_split}_${world}_1_custom ${n},${world} 4 $(( 9123+$world )) ${nodes_string}"
            full_command="pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${nodes[$n]}_${world}_${n}.log &"
            timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "${full_command}" &
            waiters+=($!)
	    #sleep 10

            #hcommand="mac_ver=0 log_path=${path_dst} ./temp_speed_reader.sh"
            #full_hcommand="pushd ${path_prefix}/aot_splitter;${hcommand} &"
            #ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "${full_hcommand}" &
            #killers+=($!)
        done
        wait ${waiters[@]}
        #clean up 
        for (( n=0;n<${#nodes[@]};n++ ))
        do
            #ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "kill -9 ${killers[$n]}; pkill -9 temp_speed; pkill -9 nw_w_comp; pkill -9 python3;"
            ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "pkill -9 temp_speed; pkill -9 nw_w_comp; pkill -9 python3; pkill -9 onnx_w_zmq.py;"
            ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
        done
	
	continue
	#skip everything down below 
        # 5 iters/batch size, 10 batch num
        waiters=()
        killers=()
        path_dst="${path_prefix}/logs/single_group_heat_con/${node_prefix}/${model_type}_${model_split}_zmq_bs_5_bn_10/${repeat}/"
        mkdir -p ${path_dst}
	for (( n=0;n<${#nodes[@]};n++ ))
        do
            command="python3 onnx_w_zmq.py 3,10 ./${model_type}_${model_split}_${world}_1_custom ${n},${world} 4 9123 ${nodes_string}"
            full_command="pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${nodes[$n]}_${world}_${n}.log &"
            timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "${full_command}" &
            waiters+=($!)

            #hcommand="mac_ver=0 log_path=${path_dst} ./temp_speed_reader.sh"
            #full_hcommand="pushd ${path_prefix}/aot_splitter;${hcommand} &"
            #ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "${full_hcommand}" &
            #killers+=($!)
        done
        wait ${waiters[@]}
        #clean up 
        for (( n=0;n<${#nodes[@]};n++ ))
        do
            #ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "kill -9 ${killers[$n]}; pkill -9 temp_speed; pkill -9 nw_w_comp; pkill -9 python3;"
            ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "pkill -9 temp_speed; pkill -9 nw_w_comp; pkill -9 python3;"
            ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
        done


    done
done
