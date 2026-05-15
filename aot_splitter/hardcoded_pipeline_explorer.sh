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
        #hardcodeed to bramble-4-2-12 here
        all_nodes=$(sinfo -N | grep "idle" | grep -v "idle\*" | grep -v "bramble-4-2-12" | grep "${node_prefix}" | awk '{print$1}')
        #node_list=( $(for i in ${all_nodes[@]}; do [[ $(cat ${path_prefix}/full_sweep/${node_prefix}_middling | grep -w "${i}" | wc -l) -gt 0 ]] && echo "${i}"; done) )
	node_list=()
        for n_check in ${all_nodes[@]}; 
	do
		# if [[ $(cat ${path_prefix}/full_sweep/${node_prefix}_middling | grep -w "${n_check}" | wc -l) -gt 0 ]]
		# if [[ $(cat ${path_prefix}/aot_splitter/${node_prefix}.model.perf | grep -w "${n_check}" | wc -l) -gt 0 ]]
		
			hname=$(ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@$n_check 'hostname')
			mem=$(ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@$n_check "echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';free | grep 'Mem'" | awk '{print $4}')
			if [[ $n_check == $hname ]]
			then
				node_list+=($n_check)
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
inter=$4
path_prefix="/home/animesh/test_model_split/"

if [[ -z $model_type ]] || [[ -z $model_split ]] || [[ -z $node_prefix ]] || [[ -z $inter ]]
# if [[ -z $model_type ]] || [[ -z $model_split ]] || [[ -z $node_prefix ]] 
then
	echo "Missing arguments, expected: ./script.sh <model_type> (resnet18, mbv3_small, eb0) <model_split> (children, modules) <node_prefix> (bramble-x-y) <split version> (int, 0(nw), 1(comp))" && exit
	# echo "Missing arguments, expected: ./script.sh <model_type> (resnet18, mbv3_small, eb0) <model_split> (children, modules) <node_prefix> (bramble-x-y)" && exit
fi


conc_val=$(sinfo -N | grep "idle" | grep -v "idle\*" | grep "${node_prefix}" | awk '{print$1}' | wc -l)

for repeat in {1..5}
# for repeat in 0
do
    # nodes=( $(node_select) )
    #hardcoded for bramble-4-2-12 to be the main runner
    # nodes=( $(sinfo -N | grep "idle" | grep -v "idle\*" | grep -v "bramble-4-2-12" | grep "${node_prefix}" | awk '{print$1}') )

    for fake_world in 6 5 4 3 2
    #only resnet and tcn for now
    do
        #filter sizes before any runs
        if [[ $( ls "${path_prefix}/aot_splitter/${model_type}_splits_${fake_world}_${inter}" | wc -l ) -eq 0 ]]
        then
            continue
        fi
        size_flag=0
        for i in "${path_prefix}/aot_splitter/${model_type}_splits_${fake_world}_${inter}/*.onnx"
        do
            size=$(ls -l ${i} | awk '{print $5}')
            if [[ $size -gt 44000000 ]]
            then
                    size_flag=1
            fi
        done
        if [[ $size_flag -eq 1 ]]
        then
            continue
        fi

        for (( fake_rank=0;fake_rank<$fake_world;fake_rank++ ))
        do
            for (( fere=0;fere<$fake_world;fere++ ))
            do
                path_dst="${path_prefix}/logs/pipeline_exploration/${node_prefix}/${model_type}_${model_split}_onnx/${fere}/${repeat}/"
                mkdir -p ${path_dst}

                #note, function is hardcoded to avoid bramble-4-2-12
                conc_val=$fere
                nodes=( $(node_select) )

                waiters=()
                killers=()
                dt=$(date -d '+60 seconds' +%s)

                #hardcoded pa rt 
                timeout 2m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@bramble-4-2-12 "pkill -9 temp_speed; pkill -9 model_rank; pkill -9 subcluster; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
                command="python3 onnxtest_w_json.py 10 ./${model_type}_splits_${fake_world}_${inter} ${fake_rank} 4 ${dt}"
                timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@bramble-4-2-12 "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${nodes[$n]}_${fake_world}_${fake_rank}.log &" &
                wait_pid=$!
                waiters+=($wait_pid)

                for (( n=0;n<${#nodes[@]};n++ ))
                do
                    timeout 2m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "pkill -9 temp_speed; pkill -9 model_rank; pkill -9 subcluster; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
                    command="python3 onnxtest_w_json.py 10 ./${model_type}_splits_${fake_world}_${inter} ${fake_rank} 4 ${dt}"
                    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${nodes[$n]}_${fake_world}_${fake_rank}.log &" &
                    wait_pid=$!
                    
                    waiters+=($wait_pid)
                done
                echo "Selected ${nodes[@]} ${world}"
                wait "${waiters[@]}"

                for (( n=0;n<${#nodes[@]};n++ ))
                do
                    timeout 2m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "pkill -9 temp_speed; pkill -9 model_rank; pkill -9 subcluster; pkill -9 python3;"
                    timeout 2m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
                done
            done

            #greater than world interference
            for fere in 1 3 5
            do
                conc_val=$(($fere+$fake_world))
                path_dst="${path_prefix}/logs/pipeline_exploration/${node_prefix}/${model_type}_${model_split}_onnx/${conc_val}/${repeat}/"
                mkdir -p ${path_dst}

                #note, function is hardcoded to avoid bramble-4-2-12
                nodes=( $(node_select) )

                waiters=()
                killers=()
                dt=$(date -d '+60 seconds' +%s)

                #hardcoded pa rt 
                timeout 2m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@bramble-4-2-12 "pkill -9 temp_speed; pkill -9 model_rank; pkill -9 subcluster; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
                command="python3 onnxtest_w_json.py 10 ./${model_type}_splits_${fake_world}_${inter} ${fake_rank} 4 ${dt}"
                timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@bramble-4-2-12 "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${nodes[$n]}_${fake_world}_${fake_rank}.log &" &
                wait_pid=$!
                waiters+=($wait_pid)

                for (( n=0;n<${#nodes[@]};n++ ))
                do
                    timeout 2m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "pkill -9 temp_speed; pkill -9 model_rank; pkill -9 subcluster; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
                    command="python3 onnxtest_w_json.py 10 ./${model_type}_splits_${fake_world}_${inter} ${fake_rank} 4 ${dt}"
                    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${nodes[$n]}_${fake_world}_${fake_rank}.log &" &
                    wait_pid=$!
                    
                    waiters+=($wait_pid)
                done
                echo "Selected ${nodes[@]} ${world}"
                wait "${waiters[@]}"

                for (( n=0;n<${#nodes[@]};n++ ))
                do
                    timeout 2m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "pkill -9 temp_speed; pkill -9 model_rank; pkill -9 subcluster; pkill -9 python3;"
                    timeout 2m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
                done
            done

        done
    done
done

