#!/bin/sh

curr_dir="$PWD"
topo_hwloc="topology.hwloc"
topo_scan="topology.scan"
bench_params=""
plot_script="plot_data.sh"

# move to the main benchmark directory if in in scripts/
if [[ $curr_dir == *"scripts"* || $curr_dir == *"results"* ]]
then
   cd .. 
   curr_dir="$PWD"
fi

# remove old executables from other builds
# not necessary with makefile dependencies
# make clean

# create results folder and subfolders if they do not already exist (occurs on first git pull)
mkdir -p "${PWD}/results/overhead"
mkdir -p "${PWD}/results/bandwidth/hh/bw"
mkdir -p "${PWD}/results/bandwidth/hh/tt"
mkdir -p "${PWD}/results/bandwidth/hd/bw"
mkdir -p "${PWD}/results/bandwidth/hd/tt"
mkdir -p "${PWD}/results/bandwidth/p2p/bw"
mkdir -p "${PWD}/results/bandwidth/p2p/tt"
mkdir -p "${PWD}/results/random_access"
mkdir -p "${PWD}/results/contention/pcie/single_gpu"
mkdir -p "${PWD}/results/contention/pcie/gpu_pair"
mkdir -p "${PWD}/results/contention/pcie/single_node"
mkdir -p "${PWD}/results/contention/qpi"
mkdir -p "${PWD}/results/contention/mem"

# compile benchmark, check for errors
# if errors, check alternate build
# exit if no working builds
echo -e "\n\e[34m\e[1mCompiling benchmark and tools...\e[0m"
error=$(make 2>&1 | grep "error")
hasError=$error
error=$(make nocpp 2>&1 | grep "error")
hasErrorNoCpp=$error

exe="run"
if [ "$hasError" != "" ]
then
   exe="run_nocpp"
   if [ "$hasErrorNoCpp" != "" ]
   then
      echo -e "\e[1m\e[31mErrors during compilation...exiting!\e[0m!"
      exit
   fi
fi

echo -e "\n\e[34m\e[1mCompilation complete!\e[0m\n"

# map system topology
echo -e "\n\e[34m\e[1mMapping system topology... ${PWD}/results/\e[0m\e[1m\e[31m$topo_hwloc\e[0m\n"

if [ -f "${PWD}/results/$topo_hwloc" ]
then
   rm ${PWD}/results/$topo_hwloc
fi

# hwloc command line topology tree stdout print
lstopo -v >> ${PWD}/results/$topo_hwloc
cat ${PWD}/results/$topo_hwloc

# check for user benchmark parameter file
if [[ $1 != '' ]]
then
   bench_params="$1"
   echo -e "\n\e[34m\e[1mInitiating benchmark (\e[1m\e[31m$exe\e[0m\e[34m\e[1m) with parameter file:\e[0m \e[1m\e[31m$bench_params\e[0m"
else
   echo -e "\n\e[34m\e[1mInitiating benchmark with default parameters:\e[0m"
fi

sleep 1

# run bandwidth benchmark
eval ./$exe $bench_params

# plot data and save graphs
echo -e "\e[34m\e[1mRunning plotting scripts:\e[0m \e[1m\e[31m$plot_script\e[0m\n"
#${PWD}/scripts/plot_data.sh


