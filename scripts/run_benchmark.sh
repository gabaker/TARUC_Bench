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

#create results folder if it does not already exist (occurs on first git pull)
if [ ! -d "${PWD}/results" ]; then
   mkdir results
fi

#compile benchmark
echo -e "\n\e[34m\e[1mCompiling benchmark and tools...\e[0m"
make all
echo -e "\e[34m\e[1mCompilation complete\e[0m\n"

#map system topology
echo -e "\e[34m\e[1mMapping system topology...\e[0m ${PWD}/results/\e[1m\e[31m$topo_file\e[0m\n"
${PWD}/scripts/map_topology.sh "./results/$topo_file"
cat ./results/$topo_file

#check for user benchmark parameter file
if [[ $1 != '' ]]
then
   bench_params="$1"
   echo -e "\n\e[34m\e[1mInitiating benchmark with parameter file:\e[0m \e[1m\e[31m$bench_params\e[0m\n"
else
   echo -e "\n\e[34m\e[1mInitiating benchmark with default parameters:\e[0m\n"
fi

# and topology_file \e[1m\e[31m$topo_file\e[0m

sleep 2

#run bandwidth benchmark
./bench $bench_params

#plot data and save graphs
echo -e "\e[34m\e[1mRunning plotting scripts:\e[0m\n"

echo -e "Script running... \e[1m\e[31m$plot_script\e[0m\n"

TopoFile="./results/topology.out"
nodes=$(sed '1!d' $TopoFile)
cpus=$(sed '2!d' $TopoFile)
gpus=$(sed '3!d' $TopoFile)

python2 ${PWD}/scripts/plot_overhead.py ${PWD}/results/results_overhead.csv $nodes $cpus $gpus

