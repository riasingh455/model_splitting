#!/bin/bash
flag=0
script_path="/home/animesh/model_splitting/full_sweep/"
pids=()
for i in {1..4}; #do 
do
	#echo "i" $i
	pids=()
	for j in {1..6}; #do
	do
		#./sweeper.sh $i $j &
		if [ -f ${script_path}/b_${i}_${j}_ml_sweep.sh ]
		then	
			/bin/bash ${script_path}/b_${i}_${j}_ml_sweep.sh &
		else
			continue
		fi
		pids+=($!) 
	done
done 

while true; do
	flag=0
	for k in ${pids[@]}; do
		if [ -d /proc/$k ]; then 
			flag=1
			break
		fi
	done
	
	if [ $flag -eq 1 ]; then
		#for c in $(squeue | grep "ReqNodeNotAvail" | awk '{print $9}'); do
		for c in $(squeue | grep "ReqNodeNotAvail" | awk '{print $9}' | awk -F':' '{print $2}' | awk -F')' '{print $1}'); do
			#we need to kill everyone related to this new failed node
			str_search=$(echo $c | awk -F'-' '{print $1"-"$2"-"$3"-"}')
			echo "KILLING ALL ${str_search}"
			for f in $(squeue | grep "${str_search}"); do
				scancel $f
			done
			#scancel $(squeue | grep "bramble")
		done
	else
		break
	fi
done
	#break


