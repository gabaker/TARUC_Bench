#!/bin/sh

if [[ "$1" != '' ]]
then
   exec >  $1
   exec 2>&1
fi

cpuinfo_dir="/proc/cpuinfo"
system_dir="/sys/devices/system"
cpu_dev_dir="$system_dir/cpu"
node_dir="$system_dir/node"


node_online="$(cat $node_dir/online)"
node_present="$(cat $node_dir/possible)"
pu_online="$(cat $cpu_dev_dir/online)"
pu_present="$(cat $cpu_dev_dir/present)"

delim_idx=`expr index "$pu_present" "-"`
min_cpu_id=${pu_online:0:(($delim_idx - 1))}
max_cpu_id=${pu_online:$(($delim_idx))}

delim_idx=`expr index "$node_present" "-"`
min_node_id=${node_online:0:(($delim_idx - 1))}
max_node_id=${node_online:$(($delim_idx))}


echo    "$(($max_cpu_id - $min_cpu_id + 1))|$(($max_node_id - $min_node_id + 1))"
echo    "------------------------------------- System Topology -------------------------------------------"
echo    "-                                                                                               -"
echo -e "-                    CPU/Socket/Node count: $((max_node_id - min_node_id + 1))\t\t\t\t\t\t        -"
echo -e "-                    CPU/Socket/Node ID list: $node_online\t\t                                -"
echo -e "-                    Min PU ID: $min_cpu_id Max PU ID: $max_cpu_id\t\t\t\t                        -"
echo    "-                    Polling Device Info from: /proc/cpuinfo                                    -"
echo    "-                                              /sys/devices/system/cpu                          -"
echo    "-                                              /sys/devices/system/cpu/cpuX/topology            -"
echo    "-                    Terms:                                                                     -"
echo    "-                          PU (Processing Unit): -A hardware thread                             -"
echo    "-                                                -For unix systems: PUs are the same as cores   -"
echo    "-                                                 on systems without hyperthreading             -"
echo    "-                                                                                               -"
echo    "-                          Core ID: Logical core ID, each CPU has unique IDs for each           -"
echo    "-                                   core present in the package/socket                          -"
echo    "-                                                                                               -"
echo    "-                          Socket ID: The logical CPU/package/node ID; identifies unique        -"
echo    "-                                     hardware CPUs rather than PUs                             -"
echo    "-                                                                                               -"
echo    "-                          Siblings/Sib: Related hardware PUs cooresponding to the number of    -"
echo    "-                                        other PUs at a given topology layer and location       -"
echo    "-                                                                                               -"
echo    "-                                                                                               -"
echo    "-------------------------------------------------------------------------------------------------"
echo    "------------------------------------ Computational Structure ------------------------------------"
echo    "-------------------------------------------------------------------------------------------------"
echo -e "- PU ID\t|Core ID| CPU Num Cores\t|   Socket ID\t|   Sib Count\t| Core Siblings\t| CPU Siblings\t-"

line_num=0
while IFS= read -r LINE; 
do
   if [[ $(echo "$LINE" | grep -e "processor" -e "core id" -e "cpu cores" -e "physical id" -e "siblings") != "" ]]
   then
      line_num=$((line_num + 1))
   fi 

   if [[ $(echo ${LINE}) == *'processor'* ]]
   then
      logical_pu=$((${LINE:$((`expr index "$LINE" ":"`))}))
      info_dir="${cpu_dev_dir}/cpu${logical_pu}/topology"
      thread_siblings="$(cat ${info_dir}/thread_siblings_list)"
      socket_siblings="$(cat ${info_dir}/core_siblings_list)"
   elif [[ $(echo ${LINE}) == *'core id'* ]] 
   then
      core_id=$((${LINE:$((`expr index "$LINE" ":"`))}))
   elif  [[ $(echo ${LINE}) == *'cpu cores'* ]]
   then
      cpu_num_cores=$((${LINE:$((`expr index "$LINE" ":"`))}))
   elif [[ $(echo ${LINE}) == *'physical id'* ]]
   then
      pu_socket_id=$((${LINE:$((`expr index "$LINE" ":"`))}))
   elif [[ $(echo ${LINE}) == *'siblings'* ]]
   then
      num_siblings=$((${LINE:$((`expr index "$LINE" ":"`))}))
   fi
   
   if [[ $line_num == 5 ]]
   then
      line_num=0
      echo -e "|   $logical_pu\t|   $core_id\t|\t$cpu_num_cores\t|\t$pu_socket_id\t|\t$num_siblings\t|     $thread_siblings\t|  $socket_siblings\t|"
   fi
done < $cpuinfo_dir

echo    "-------------------------------------------------------------------------------------------------"
echo    "--------------------------------------- Memory Hierarchy ----------------------------------------"
echo    "-                                                                                               -"
echo    "-------------------------------------------------------------------------------------------------"
