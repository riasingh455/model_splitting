#!/bin/bash
temp_log_path="/home/animesh/test_model_split/logs"
if [ -z $log_path ]
then
	log_path=$temp_log_path
fi
if [[ $mac_ver -eq 1 ]]
then
    t=$mac_suffix
else
    t=$(hostname)
fi
while :
    do
	    if [[ $mac_ver -eq 1 ]]
        then
            echo -e "$(date '+TIME:%H:%M:%S')" >> ${log_path}/speed_heat$t.log
            echo "cpu:$(uptime | awk '{print $10, $11, $12}')" >> ${log_path}/speed_heat$t.log
        else
            echo -e "$(date '+TIME:%H:%M:%S')\n$(cat /sys/class/thermal/thermal_zone0/temp)" >> ${log_path}/speed_heat$t.log
            echo "cpu:$(uptime | awk '{print $10, $11, $12}'),freq:$(vcgencmd measure_clock arm),throttle_flag:$(vcgencmd get_throttled),voltage:$(vcgencmd measure_volts core),$(vcgencmd measure_volts sdram_c),$(vcgencmd measure_volts sdram_i),$(vcgencmd measure_volts sdram_p)" >> ${log_path}/speed_heat$t.log
        fi
        sleep 1
    done
