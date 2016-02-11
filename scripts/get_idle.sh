#!/bin/sh

state=0

while [ $state -lt 5 ]
do
   echo "State${state}"
   cpu=0
   while [ $cpu -lt 32 ]
   do
      echo "CPU$cpu $(cat /sys/devices/system/cpu/cpu${cpu}/cpuidle/state${state}/time)"
      cpu=$((cpu+1))
   done
   state=$((state+1))
done
