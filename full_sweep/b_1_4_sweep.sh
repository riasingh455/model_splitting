#!/bin/bash
#middling_devs=(2 4 6 8 10 12 13 16 22 24 27 28 31 32 34 36 38 40)
middling_devs=(3 4 5 7 10 11 12 14 15 16 17 19 21 23 27 28 29 30 31 32 33 35 36 37 38 39 40 41 42)
fragile_devs=(2 6 8 9 13 18 20 24 25 26)
#log_path="/home/animesh/full_sweep_logs/fragile_rank/b-${1}-${2}/"
log_path="/home/animesh/full_sweep_logs/full_rank/b-${1}-${2}/"
mkdir -p ${log_path}/600_logs

script_path="/home/animesh/scripts/"

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
 size=$1
 sample=$2
 i=$3
 val=$4
        if [ $(sinfo -N | grep ${val}-${i} | grep "idle" | grep -v "idle\*" | wc -l) -eq 0 ]
        then
                mkdir -p ${log_path}/${i}_${size}_${sample}_logs
                mv ${log_path}/${size}_logs/* ${log_path}/${i}_${size}_${sample}_logs/

                continue
        fi

        #randomly sample - sample number of devices
        visited=($i)
        devs=()
        while [ ${#devs[@]} -lt $sample ] && [ ${#visited[@]} -lt ${#middling_devs[@]} ]
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



        log_path=$log_path start=4 end=4 size=$1 srun -N 1 --nodelist="${val}-${i}" ${script_path}/volt_tester.sh 4 0  > /dev/null&
        pids=($!)

	for v in ${devs[@]}
        do
                log_path=$log_path start=4 end=4 size=$1 srun -N 1 --nodelist="${val}-${v}" ${script_path}/volt_tester.sh 4 0  > /dev/null&
                pids+=($!)
        done

        echo "starting now for ${i} with devices ${devs[@]}"
        wait ${pids[@]}

        mkdir -p ${log_path}/${i}_${size}_${sample}_logs
        mv ${log_path}/${size}_logs/* ${log_path}/${i}_${size}_${sample}_logs/
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

range_start=(4 22 22 24 24 4 24 4 14 22 24 4 14 14 4 24 4 4 0 4 14 0 4 4 14 14 14 0 14)
range_end=(13 28 28 28 28 13 28 13 23 28 28 13 23 23 13 28 13 13 5 13 23 5 13 13 23 23 23 5 23)

pdu_tag="bramble-${1}-${2}"

#for run in {1..30};
#for run in 1;
#do
#        for (( ind=0; ind<${#middling_devs[@]}; ind++ ));
#        do
           #for (( s=${range_start[$ind]}; s<=${range_end[$ind]}; s++ ));
        #  for s in 0 1 5 10
           #do
#                specific_fragile_test 600 ${middling_devs[$ind]} $pdu_tag
           #done
#        done
#        mkdir -p ${log_path}/${run}_ranking_logs/
#        mv ${log_path}/*_600_*_logs ${log_path}/${run}_ranking_logs/
#done

#exit


for run in {1..30};
#for run in 1;
do
	for (( ind=0; ind<${#middling_devs[@]}; ind++ ));
        do
		for (( s=${range_start[$ind]}; s<=${range_end[$ind]}; s++ ));
        	#for s in 0 1 10 20 28
		do
	      		specific_middling_test 600 $s ${middling_devs[$ind]} $pdu_tag
		done
        done
        mkdir -p ${log_path}/${run}_ranking_logs/
        mv ${log_path}/*_600_*_logs ${log_path}/${run}_ranking_logs/
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

