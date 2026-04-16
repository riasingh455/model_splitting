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
		if [[ $(cat ${path_prefix}/full_sweep/${node_prefix}_middling | grep -w "${n_check}" | wc -l) -gt 0 ]]
		then
			hname=$(ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@$n_check 'hostname')
			mem=$(ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@$n_check "echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';free | grep 'Mem'" | awk '{print $4}')
			if [[ $n_check == $hname ]] && [[ $mem -ge 650000 ]]
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
bnum=$4
path_prefix="/home/animesh/test_model_split/"

if [[ -z $model_type ]] || [[ -z $model_split ]] || [[ -z $node_prefix ]] || [[ -z $bnum ]]
then
	echo "Missing arguments, expected: ./script.sh <model_type> (resnet18, mbv3_small, eb0) <model_split> (children, modules) <node_prefix> (bramble-x-y) <number of batchs> (int)" && exit
fi

highest_flop_rank=( 0 0 1 1 2 )
# for repeat in 3{1..10}
for repeat in 1
do
    #for fake_world in 2 3 4 5
    for fake_world in 3 4 5 6 7 8
    do
        # for (( conc_val=0; conc_val<$fake_world; conc_val++ ))
        # do
            conc_val=$fake_world
            nodes=( $(node_select) )
        #     fake_rank_idx=$(( $fake_world-1 ))
        #     fake_rank=${highest_flop_rank[$fake_rank_idx]}
            
            path_dst="${path_prefix}/logs/group_heat/${node_prefix}/${model_type}_${model_split}/${conc_val}_concurrency_level/${repeat}/"
            mkdir -p ${path_dst}
            waiters=()
            for (( n=0;n<${#nodes[@]};n++ ))
            do
	    	ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "pkill -9 roofline; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"
                # timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "pushd $path_prefix; node_prefix=$node_prefix world=$conc_val rank=$n master=${nodes[0]} model_type=$model_type model_split=$model_split bnum=$bnum ./roofline_pi_script_no_slurm_no_temp.sh &" &
		command="python3 single_runner.py --cores 4 --rank ${n} --world ${fake_world} --ip ${nodes[0]} --port 8123 --warmup 1 --batch-size 1 --batch-num ${bnum} --iters 10 --model-type ${model_type} --model-split-type ${model_split}"
                command="${command} --fake_rank ${n} --fake_world ${fake_world}"
                timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${nodes[$n]} &" &
                waiters+=($!)
            done
            echo "Selected ${nodes[@]} ${world}"
    		wait "${waiters[@]}"
            #cleanup
            for (( n=0;n<${#nodes[@]};n++ ))
            do
	    		ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "pkill -9 roofline; pkill -9 python3; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';"			    
            done

        # done
    done
done
