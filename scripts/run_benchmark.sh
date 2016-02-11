#!/bin/sh

curr_dir="$PWD"
topo_file="topology.in"
bench_params=""
plot_script="plot_overhead.py"


#move to the main benchmark directory if in in scripts/
if [[ $curr_dir == *"scripts"* ]]
then
   cd .. 
fi

#compile benchmark
echo -e "\n\e[34m\e[1mCompiling benchmark and tools...\e[0m"
make all
echo -e "\e[34m\e[1mCompilation complete\e[0m\n"

#map system topology
echo -e "\e[34m\e[1mMapping system topology...\e[0m ${PWD}/\e[1m\e[31m$topo_file\e[0m\n"
${PWD}/scripts/map_topology.sh "$topo_file"
cat $topo_file

#check for user benchmark parameter file
if [[ $1 != '' ]]
then
   bench_params="$1"
   echo -e "\n\e[34m\e[1mInitiating benchmark with:\e[0m parameter_file \e[1m\e[31m$bench_params\e[0m and topology_file \e[1m\e[31m$topo_file\e[0m\n"
else
   echo -e "\n\e[34m\e[1mInitiating benchmark with: default parameters and topology \e[1m\e[31m$topo_file\e[0m\n"
fi

sleep 2

#run bandwidth benchmark
./bench $bench_params


#plot data and save graphs
echo -e "\e[34m\e[1mTo plot benchmark data use the python plotting script: \e[1m\e[31m$plot_script\e[0m\n"

