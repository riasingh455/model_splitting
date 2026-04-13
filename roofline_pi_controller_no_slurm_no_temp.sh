#!/bin/bash 

rng_gen () {
        len=$1
        rng_arr=()
        while [[ ${#rng_arr[@]} -lt $world ]]
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
			mem=$(ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@$n_check "free | grep 'Mem'" | awk '{print $4}')
			if [[ $n_check == $hname ]] && [[ $mem -ge 500000 ]]
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
path_prefix="/home/animesh/test_model_split/"

if [[ -z $model_type ]] || [[ -z $model_split ]] || [[ -z $node_prefix ]]
then
	echo "Missing arguments, expected: ./script.sh <model_type> (resnet18, mbv3_small, eb0) <model_split> (children, modules) <node_prefix> (bramble-x-y)" && exit
fi

for repeat in {1..10}
do
	for world in {2..10}
	do
    		nodes=( $(node_select) )
		#echo ${nodes[@]}
		#exit
    		counter=0
		waiters=()
    		for n in ${nodes[@]}
    		do
         		#node_prefix=$node_prefix world=$world rank=$counter master=${nodes[0]} model_type=$model_type model_split=$model_split srun --nodelist=$n roofline_pi_script_no_slurm_no_temp.sh &
			#mem=$(ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${n} "free -h; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; free -h")
			mem=$(ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${n} "pkill -9 roofline; echo 'race4fun' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches';")
			#echo ${n} ${mem}
         		
			ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${n} "pushd $path_prefix; node_prefix=$node_prefix world=$world rank=$counter master=${nodes[0]} model_type=$model_type model_split=$model_split ./roofline_pi_script_no_slurm_no_temp.sh &" &
			waiters+=($!)
			counter=$(( $counter+1 ))
			#echo $n, $world
    		done
    		echo "Selected ${nodes[@]} ${world}"
    		wait "${waiters[@]}"
                mkdir -p ${path_prefix}/logs/roofline/${node_prefix}/${model_type}_${model_split}/${repeat}/
                mv ${path_prefix}/logs/roofline/${node_prefix}/${model_type}_${model_split}/${world}_size ${path_prefix}/logs/roofline/${node_prefix}/${model_type}_${model_split}/${repeat}/
    		#break
	done
done
