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
        node_list=( $(for i in ${all_nodes[@]}; do [[ $(cat ${path_prefix}/full_sweep/${node_prefix}_middling | grep -w "${i}" | wc -l) -gt 0 ]] && echo "${i}"; done) )
        ind_arr=( $(rng_gen "${#node_list[@]}") )
        node_select=( $(for i in ${ind_arr[@]}; do echo "${node_list[$i]}"; done) )
        echo "${node_select[@]}"
}

model_type=$1
model_split=$2
node_prefix=$3
path_prefix="/home/animesh/model_splitting/"

if [[ -z $model_type ]] || [[ -z $model_split ]] || [[ -z $node_prefix ]]
then
	echo "Missing arguments, expected: ./script.sh <model_type> (resnet18, mbv3_small, eb0) <model_split> (children, modules) <node_prefix> (bramble-x-y)" && exit
fi

for repeat in {1..10}
do
	for world in {2..10}
	do
    		nodes=( $(node_select) )
    		counter=0
    		for n in ${nodes[@]}
    		do
         		world=$world rank=$counter master=${nodes[0]} model_type=$model_type model_split=$model_split srun --nodelist=$n roofline_pi_script.sh &
	 		counter=$(( $counter+1 ))
			#echo $n, $world
    		done
    		echo "Selected ${nodes[@]} ${world}"
    		wait
                mkdir -p ${path_prefix}/logs/roofline/${model_type}_${model_split}/${repeat}/
                mv ${path_prefix}/logs/roofline/${model_type}_${model_split}/${world}_size ${path_prefix}/logs/roofline/${model_type}_${model_split}/${repeat}/
    		#break
	done
done
