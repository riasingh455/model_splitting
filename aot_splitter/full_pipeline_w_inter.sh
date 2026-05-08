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

p_n () {
    counter=0
    for (( end=1;end<=$pipeline_n;end++ ))
    do
        waiters=()
        path_dst="${path_prefix}/logs/full_pipeline_inter/${node_prefix}/${model_type}_${model_split}_onnx/${counter}_${fere}/${repeat}/"
        mkdir -p ${path_dst}
        counter=$(( $counter + 1 ))
	
	for inter_n in ${inter_nodes[@]}
        do
             timeout 30s ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${inter_n} "pkill -9 temp_speed; pkill -9 pipeline; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
        done
	#-----------------------------
        dt=$(date -d '+60 seconds' +%s)
	for (( start=0;start<$end;start++ ))
        do
            timeout 30s ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$start]} "pkill -9 temp_speed; pkill -9 pipeline; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
            #command="python3 onnxtest.py 10 ./${model_type}_${model_split}_${fake_world}_1_custom ${start} 4 ${dt}"
	    command="python3 onnxtest_w_json.py 10 ./${model_type}_splits_${fake_world}_${inter} ${start} 4 ${dt}"
            timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$start]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${nodes[$start]}_${fake_world}_${start}_${inter}.log &" &
            waiters+=($!)
        done
        #-----------------------------
	for inter_n in ${inter_nodes[@]}
        do
            command="python3 onnxtest_w_json.py 10 ./tcn_splits_1_0 0 4 ${dt}"
            timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${inter_n} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${inter_n}_${fake_world}_${start}_inter_${inter}.log &" &
            waiters+=($!)
        done

        wait ${waiters[@]}
    done

    for (( end=1;end<$pipeline_n;end++ ))
    do
        waiters=()
        path_dst="${path_prefix}/logs/full_pipeline_inter/${node_prefix}/${model_type}_${model_split}_onnx/${counter}_${fere}/${repeat}/"
        mkdir -p ${path_dst}
        counter=$(( $counter+1 ))
        
	
	for inter_n in ${inter_nodes[@]}
        do
                timeout 30s ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${inter_n} "pkill -9 temp_speed; pkill -9 pipeline; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
        done


	#-----------------
        dt=$(date -d '+60 seconds' +%s)
	for (( start=$end;start<5;start++ ))
        do
            timeout 30s ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$start]} "pkill -9 temp_speed; pkill -9 pipeline; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
            #command="python3 onnxtest.py 10 ./${model_type}_${model_split}_${fake_world}_1_custom ${start} 4 ${dt}"
            command="python3 onnxtest_w_json.py 10 ./${model_type}_splits_${fake_world}_${inter} ${start} 4 ${dt}"
            timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$start]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${nodes[$start]}_${fake_world}_${start}_${inter}.log &" &
            waiters+=($!)
        done
	#-----------------------
	
	for inter_n in ${inter_nodes[@]}
        do
                command="python3 onnxtest_w_json.py 10 ./tcn_splits_1_0 0 4 ${dt}"
                timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${inter_n} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${inter_n}_${fake_world}_${start}_inter_${inter}.log &" &
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
#for repeat in {1..10}
for repeat in 0
do
    #for fake_world in 5 2
    for fake_world in 2 3 4 5 6
    do
        for inter in 0 1
        do
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
            
            for fere in 0 1 3 5
            do
                conc_val=$(($fake_world+$fere))
                full_nodes=( $(node_select) )
                nodes=()
                inter_nodes=()
                for (( inter_i=0;inter_i<$fere; inter_i++ ))
                do
                    inter_nodes+=(${full_nodes[$inter_i]})
                done
                for (( inter_i=$fere;inter_i<$conc_val; inter_i++ ))
                do
                    nodes+=(${full_nodes[$inter_i]})
                done

                pipeline_n=$fake_world
                p_n 
            done
        done
    done
done

