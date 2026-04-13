#!/bin/bash

for i in $(sinfo -N | grep "${1}" | grep "idle" | grep -v "idle\*" | awk '{print $1}'); do 
	srun --nodelist=$i pkill -9 python3; 
	#srun --nodelist=$i pkill -9 python3; 
	srun --nodelist=$i pkill -9 roofline; 
done;

