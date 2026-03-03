#!/bin/bash

log_path="/home/pi/model_splitting/logs/b-${1}-${2}"
mkdir -p ${log_path}/trial
script_path="/home/pi/model_splitting/"

#run torchrun through slurm, specify right commands and push output to files
subcluster="bramble-${1}-${2}"
sinfo -N | grep "${subcluster}" | grep "idle" | grep -v "idle\*" | awk '{print $1}' > ${log_path}/machine_samples

a=$(cat ${log_path}/machine_samples)
good_ones=($a)
world=4
master=${good_ones[0]}
#pick master from choice maybe?
#currently just first in line

for (( i = 0 ; i < ${#good_ones[@]}; i++ ))
do
        val=${good_ones[$i]}
        log_path="${log_path}/trial/" world_size=$world rank=$i master=$master cores=4 srun -N 1 --nodelist=$val ${script_path}/volt_tester_ml.sh 4 0  > /dev/null&
        #wait
        if [ $i -ge $(( $world - 1 )) ]
        then
                echo "${master} and ${world}"
                #would be replaced by a wait here instead later?
                break
        fi
done
wait



