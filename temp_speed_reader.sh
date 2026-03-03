#!/bin/bash
temp_log_path="/home/pi/model_splitting/logs"
if [ -z $log_path ]
then
	log_path=$temp_log_path
fi

t=$(hostname)
while :
    do
        echo -e "$(date '+TIME:%H:%M:%S')\n$(cat /sys/class/thermal/thermal_zone0/temp)" >> ${log_path}/speed_heat$t.log
	echo "freq:$(vcgencmd measure_clock arm),throttle_flag:$(vcgencmd get_throttled),voltage:$(vcgencmd measure_volts core),$(vcgencmd measure_volts sdram_c),$(vcgencmd measure_volts sdram_i),$(vcgencmd measure_volts sdram_p)" >> ${log_path}/speed_heat$t.log
	sleep 1
    done
