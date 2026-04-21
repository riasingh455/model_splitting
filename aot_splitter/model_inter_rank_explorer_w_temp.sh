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


#conc_val=5

for repeat in {1..10}
#for repeat in 0
do
    #nodes=( $(node_select) )
    # for fake_world in 5 10 15
    # for fake_world in 15
    for fake_world in 5 2
    do
	conc_val=$fake_world
    	nodes=( $(node_select) )

	for wait_flag in 1 0
        #for (( n=0;n<${#nodes[@]};n++ ))
        #for (( fake_rank=0;fake_rank<$fake_world;fake_rank++ ))
        # for fake_rank in 4 13
        # for fake_rank in 0
        do
                #fake_rank_idx=$(( $fake_world-1 ))
                # #fake_rank=${highest_flop_rank[$fake_rank_idx]}
                # fake_rank=${highest_mem_rank[$fake_rank_idx]}
                path_dst="${path_prefix}/logs/single_group_heat_con/${node_prefix}/${model_type}_${model_split}_onnx_inter_rank/all_dev/${repeat}/"
                mkdir -p ${path_dst}
                waiters=()
                killers=()
		#for wait_flag in 1 0
        	#for (( n=0;n<${#nodes[@]};n++ ))
		#do
        	dt=$(date -d '+60 seconds' +%s)
		if [[ $wait_flag -eq 1 ]]
		then
			dt=$(date -d '+1 seconds' +%s)
			path_dst="${path_prefix}/logs/single_group_heat_con/${node_prefix}/${model_type}_${model_split}_onnx_inter_rank/per_dev/${repeat}/"
			mkdir -p ${path_dst}
		fi
        		
        	for (( n=0;n<${#nodes[@]};n++ ))
		#for (( fake_rank=0;fake_rank<$fake_world;fake_rank++ ))
               	#for (( n=0;n<${#nodes[@]};n++ ))
                do
                    	ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "pkill -9 temp_speed; pkill -9 model_rank; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
                    	# timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "pushd $path_prefix; node_prefix=$node_prefix world=$conc_val rank=$n master=${nodes[0]} model_type=$model_type model_split=$model_split bnum=$bnum ./roofline_pi_script_no_slurm_no_temp.sh &" &
                    	# command="python3 single_runner.py --custom --cores ${cores} --rank ${n} --world ${conc_val} --ip ${nodes[0]} --port 8123 --warmup 1 --batch-size 4 --batch-num 1 --iters 10 --model-type ${model_type} --model-split-type ${model_split}"
                    	# onnxtest.py 10 ./vit_modules_3_1_custom 0 4 ${dt}
                    		
			hcommand="mac_ver=0 log_path=${path_dst} ./temp_speed_reader.sh"
                    	full_hcommand="pushd ${path_prefix}/aot_splitter;${hcommand} &"
                    		
			command="python3 onnxtest.py 10 ./${model_type}_${model_split}_${fake_world}_1_custom ${n} 4 ${dt}"
                    	# command="${command} --fake_rank ${fake_rank} --fake_world ${fake_world}"
                    	timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${nodes[$n]}_${fake_world}_${n}.log &" &
                    	wait_pid=$!
			ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "${full_hcommand}" &
                    	kill_pid=$!
			if [[ $wait_flag -eq 1 ]]
                    	then
                    	     wait $wait_pid
			     kill -9 $kill_pid
			     #continue
                    	fi
                    	waiters+=($wait_pid)
                    	killers+=($kill_pid)
                done
                echo "Selected ${nodes[@]} ${world}"
		if [[ $wait_flag -eq 0 ]]
		then
                	wait "${waiters[@]}"
		fi
		#cleanup
                for (( n=0;n<${#nodes[@]};n++ ))
                do
			if [[ $wait_flag -eq 0 ]]
			then
				ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "kill -9 ${killers[$n]};"
			fi
                  	ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "pkill -9 temp_speed; pkill -9 model_rank; pkill -9 python3;"
                    	# $(tail -n 1 ${path_dst}/speed_chronos${nodes[$n]}_${fake_rank}.log) > ${path_dst}/speed_journal${nodes[$n]}_${fake_rank}.log
                    	ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
                done
        done
    done
done

