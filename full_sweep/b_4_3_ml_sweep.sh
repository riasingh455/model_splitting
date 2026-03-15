#!/bin/bash
#middling_devs=(2 4 6 8 10 12 13 16 22 24 27 28 31 32 34 36 38 40)
middling_devs=(1 2 3 4 7 8 9 10 11 12 13 14 16 18 19 20 22 24 25 26 27 28 29 31 34 36 37 38 39 40 41 42)

log_path="/home/animesh/model_splitting/logs/sweep/b-4-3/"
mkdir -p ${log_path}

script_path="/home/animesh/model_splitting/"

specific_fragile_test () {

 size=$1
 i=$2
 val=$3
 if [ $(sinfo -N | grep "${val}-${i}" | grep "idle" | grep -v "idle\*" | wc -l) -eq 0 ]
 then
	mkdir -p ${log_path}/${i}_${size}_${sample}_logs
        mv ${log_path}/${size}_logs/* ${log_path}/${i}_${size}_${sample}_logs/
	continue
 fi
 visited=("${val}-${i}")
 for f in ${fragile_devs[@]}
 do
	 if [ $(sinfo -N | grep "${f}" | grep "idle" | grep -v "idle\*" | wc -l) -eq 0 ]
         then
               continue
	 fi
	 visited+=($f)
 done

 pids=()

 for v in ${visited[@]}
 do
     log_path=$log_path start=4 end=4 size=$1 srun -N 1 --nodelist="${val}-${v}" ${script_path}/volt_tester.sh 4 0  > /dev/null&
     pids+=($!)
 done

 echo "starting now for ${i} with devices ${visited[@]}"
 wait ${pids[@]}
 mkdir -p ${log_path}/${i}_${size}_${sample}_logs
 mv ${log_path}/${size}_logs/* ${log_path}/${i}_${size}_${sample}_logs/
}

specific_middling_test () {
 world=$1
 batch_size=$2
 batch_num=$3
 rank=0
 #size=$1
 #sample=$2
 i=$4
 val=$5
 iters=$6
 folder_key="run/${i}_${world}_${batch_size}_${batch_num}"
        if [ $(sinfo -N | grep -w "${val}-${i}" | grep "idle" | grep -v "idle\*" | wc -l) -eq 0 ]
        then
                #mkdir -p ${log_path}/${folder_key}_logs
                #mv ${log_path}/${size}_logs/* ${log_path}/${i}_${size}_${sample}_logs/

                return
        fi
	echo "Node ${val}-${i} Available"
        #randomly sample - sample number of devices
        visited=($i)
        devs=()
	while [ ${#devs[@]} -lt $(( $world-1 )) ] && [ ${#visited[@]} -lt ${#middling_devs[@]} ]
        do
                d=${middling_devs[ $RANDOM % ${#middling_devs[@]} ]}

                flag=0
                for v in ${visited[@]}
                do
                        if [ $v -eq $d ]
                        then
                                flag=1
                                break
                        fi
                done

                if [ $flag -eq 1 ]
                then
                        continue
                fi

                visited+=($d)
                if [ $(sinfo -N | grep "${val}-${d}" | grep "idle" | grep -v "idle\*" | wc -l) -eq 0 ]
                then
                        continue
                fi
                devs+=($d)
        done



        #log_path=$log_path start=4 end=4 size=$1 srun -N 1 --nodelist="${val}-${i}" ${script_path}/volt_tester.sh 4 0  > /dev/null&
        mkdir -p "${log_path}/${folder_key}_logs/"
        iters=$iters log_path="${log_path}/${folder_key}_logs/" batch_num=$batch_num batch_size=$batch_size world_size=$world rank=$rank master="${val}-${i}" cores=1 srun -N 1 --nodelist="${val}-${i}" ${script_path}/volt_tester_ml.sh 4 0  > /dev/null&
        pids=($!)
        #counter=1
	for v in ${devs[@]}
        do
                rank=$(( $rank + 1 ))
                #counter=$(( $counter + 1 ))
                #log_path=$log_path start=4 end=4 size=$1 srun -N 1 --nodelist="${val}-${v}" ${script_path}/volt_tester.sh 4 0  > /dev/null&
                iters=$iters log_path="${log_path}/${folder_key}_logs/" batch_num=$batch_num batch_size=$batch_size world_size=$world rank=$rank master="${val}-${i}" cores=1 srun -N 1 --nodelist="${val}-${v}" ${script_path}/volt_tester_ml.sh 4 0  > /dev/null&
                pids+=($!)
        done

        echo "starting now for ${i} with devices ${devs[@]}"
        wait ${pids[@]}

        #mkdir -p ${log_path}/${i}_${size}_${sample}_logs
        #mv ${log_path}/${size}_logs/* ${log_path}/${i}_${size}_${sample}_logs/
}


full_mm_stress () {
#individually stress different devices with different mm_sizes

sinfo -N | grep "${2}" | grep "idle" | grep -v "idle\*" | awk '{print $1}' > ${log_path}/machine_samples
a=$(cat ${log_path}/machine_samples)
good_ones=($a)
for (( i = 0 ; i < ${#good_ones[@]}; i++ ))
do
        val=${good_ones[$i]}
        log_path=${log_path} start=4 end=4 size=$1 srun -N 1 --nodelist=$val ${script_path}/volt_tester.sh 4 0  > /dev/null&
        #wait
done
wait

}

ind_mm_stress () {
#individually stress different devices with different mm_sizes

sinfo -N | grep "${2}" | grep "idle" | grep -v "idle\*" | awk '{print $1}' > ${log_path}/machine_samples
a=$(cat ${log_path}/machine_samples)
good_ones=($a)
for (( i = 0 ; i < ${#good_ones[@]}; i++ ))
do
        val=${good_ones[$i]}
        log_path=${log_path} start=4 end=4 size=$1 srun -N 1 --nodelist=$val ${script_path}/volt_tester.sh 4 0  > /dev/null&
        wait
done

} 

#sinfo -N | grep "bramble-4-5-" | grep "idle" | grep -v "idle\*" | awk '{print $1}' > ${log_path}/machine_samples
#a=$(cat ${log_path}/machine_samples)
#good_ones=($a)
#clock_devs ${good_ones[@]}
#(2 4 5 7 13 14 26 27 31 33 38)
#range_start=(6 6 0 4 4 6 0 0 4 6 6)
#range_end=(10 10 5 10 10 10 8 8 10 10 10)

pdu_tag="bramble-1-2"


#for run in {1..30};
#for run in 1;
#do
#	for (( ind=0; ind<${#middling_devs[@]}; ind++ ));
#	do
	   #for (( s=${range_start[$ind]}; s<=${range_end[$ind]}; s++ ));
	#  for s in 0 1 5 10
	   #do
#                specific_fragile_test 600 ${middling_devs[$ind]} $pdu_tag
	   #done
#	done
#	mkdir -p ${log_path}/${run}_ranking_logs/
#	mv ${log_path}/*_600_*_logs ${log_path}/${run}_ranking_logs/
#done

#exit

#for run in {1..30};
#for run in 1;
#do
world_sizes=(4 2 1) #resnet-18 max splits is 6 -> empirically found, but only 4 can run reliably
batches=(4 1)
num_batches=(5 3 1)
for run in 1; #{1..10};
do 
        #each middling device has 30 selections, 10 total model iterations (5 counted for avg)
        #each world size tests different devices every iteration of the 30 we check here
        #extremes check differnt device and pipeline overlaps -> more compute intensive? less compute intensive?
	mkdir -p ${log_path}/run/
        for (( ind=0; ind<${#middling_devs[@]}; ind++ ));
        do
	        for (( s=0; s<${#world_sizes[@]}; s++ ));
        	        #for s in 0 1 5 10
	        do
	                for (( b=0; b<${#batches[@]}; b++ ));
	                do
	                        for (( n=0; n<${#num_batches[@]}; n++ ));
	                        do
	                                echo "STARTED RUN ${world_sizes[$s]} ${batches[$b]} ${num_batches[$n]} ${middling_devs[$ind]}"
					specific_middling_test ${world_sizes[$s]} ${batches[$b]} ${num_batches[$n]} ${middling_devs[$ind]} $pdu_tag 5  #600 $s ${middling_devs[$ind]} $pdu_tag
	                                echo "FINISHED RUN ${world_sizes[$s]} ${batches[$b]} ${num_batches[$n]} ${middling_devs[$ind]}"
	                        done
	                done
	        done
		#break
        done
        
        mkdir -p ${log_path}/${run}_ranking_logs/
        mv ${log_path}/run/* ${log_path}/${run}_ranking_logs/
        #mv ${log_path}/*_600_*_logs ${log_path}/${run}_ranking_logs/
done

exit


for run in 1: #{1..30};
do
        #full_mm_stress 600
        ind_mm_stress 600 "bramble-${1}-${2}"
        mkdir -p ${log_path}/ind_${run}_600
        mv ${log_path}/600_logs/* ${log_path}/ind_${run}_600/
done

sleep 60

for run in 1: #{1..30};
do
        full_mm_stress 600 "bramble-${1}-${2}"
        #ind_mm_stress 600
        mkdir -p ${log_path}/all_${run}_600
        mv ${log_path}/600_logs/* ${log_path}/all_${run}_600/
done


exit


for run in {1..30}
do
	for group in 1 10 20
	do
		throttle_support 600 $group $run 
	done
done

#clock_everyone ${not_bad_devs[@]}

