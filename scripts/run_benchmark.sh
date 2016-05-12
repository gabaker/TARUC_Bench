#!/bin/sh

curr_dir="$PWD"
topo_file="topology.in"
bench_params=""
plot_script="plot_data.sh"

#move to the main benchmark directory if in in scripts/
if [[ $curr_dir == *"scripts"* ]]
then
   cd .. 
fi

#create results folder and subfolders if they do not already exist (occurs on first git pull)
if [ ! -d "${PWD}/results" ]; then
   mkdir "${PWD}/results"
fi

if [ ! -d "${PWD}/results/overhead" ]; then
   mkdir "${PWD}/results/overhead"
fi

if [ ! -d "${PWD}/results/bandwidth" ]; then
   mkdir "${PWD}/results/bandwidth"
fi

if [ ! -d "${PWD}/results/contention" ]; then
   mkdir "${PWD}/results/congestion"
fi

if [ ! -d "${PWD}/results/usage" ]; then
   mkdir "${PWD}/results/usage"
fi


#compile benchmark
echo -e "\n\e[34m\e[1mCompiling benchmark and tools...\e[0m"
make
echo -e "\e[34m\e[1mCompilation complete\e[0m\n"

#map system topology
echo -e "\e[34m\e[1mMapping system topology...\e[0m ${PWD}/results/\e[1m\e[31m$topo_file\e[0m\n"
#${PWD}/scripts/map_topology.sh "./results/$topo_file"
#cat ./results/$topo_file
lstopo -v

#check for user benchmark parameter file
if [[ $1 != '' ]]
then
   bench_params="$1"
   echo -e "\n\e[34m\e[1mInitiating benchmark with parameter file:\e[0m \e[1m\e[31m$bench_params\e[0m\n"
else
   echo -e "\n\e[34m\e[1mInitiating benchmark with default parameters:\e[0m\n"
fi

sleep 2

#run bandwidth benchmark
./run $bench_params

#plot data and save graphs
echo -e "\e[34m\e[1mRunning plotting scripts:\e[0m \e[1m\e[31m$plot_script\e[0m\n"
./${PWD}/scripts/plot_data.py




