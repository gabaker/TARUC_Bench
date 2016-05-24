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
if [ ! -d "${PWD}/results" ]
then
   mkdir "${PWD}/results"
fi

if [ ! -d "${PWD}/results/overhead" ]
then
   mkdir "${PWD}/results/overhead"
fi

if [ ! -d "${PWD}/results/bandwidth" ]
then
   mkdir "${PWD}/results/bandwidth"
   mkdir "${PWD}/results/bandwidth/hh"
   mkdir "${PWD}/results/bandwidth/hh/bw"
   mkdir "${PWD}/results/bandwidth/hh/tt"
   mkdir "${PWD}/results/bandwidth/hd"
   mkdir "${PWD}/results/bandwidth/hd/bw"
   mkdir "${PWD}/results/bandwidth/hd/tt"
   mkdir "${PWD}/results/bandwidth/p2p"
   mkdir "${PWD}/results/bandwidth/p2p/bw"
   mkdir "${PWD}/results/bandwidth/p2p/tt"
else
   if [ ! -d "${PWD}/results/bandwidth/hh" ]
   then
      mkdir "${PWD}/results/bandwidth/hh"
      mkdir "${PWD}/results/bandwidth/hh/bw"
      mkdir "${PWD}/results/bandwidth/hh/tt"
   else
      if [ ! -d "${PWD}/results/bandwidth/hh/bw" ]
      then
         mkdir "${PWD}/results/bandwidth/hh/bw"
      fi
      
      if [ ! -d "${PWD}/results/bandwidth/hh/tt" ]
      then
         mkdir "${PWD}/results/bandwidth/hh/tt"
      fi

   fi

   if [ ! -d "${PWD}/results/bandwidth/hd" ]
   then
      mkdir "${PWD}/results/bandwidth/hd"
      mkdir "${PWD}/results/bandwidth/hd/bw"
      mkdir "${PWD}/results/bandwidth/hd/tt"
   else 
      if [ ! -d "${PWD}/results/bandwidth/hd/bw" ] 
      then
         mkdir "${PWD}/results/bandwidth/hd/bw"
      fi
      
      if [ ! -d "${PWD}/results/bandwidth/hd/tt" ]
      then
         mkdir "${PWD}/results/bandwidth/hd/tt"
      fi
   fi

   if [ ! -d "${PWD}/results/bandwidth/p2p" ]
   then
      mkdir "${PWD}/results/bandwidth/p2p"
      mkdir "${PWD}/results/bandwidth/p2p/bw"
      mkdir "${PWD}/results/bandwidth/p2p/tt"
   else
      if [ ! -d "${PWD}/results/bandwidth/p2p/bw" ]
      then
         mkdir "${PWD}/results/bandwidth/p2p/bw"
      fi
      
      if [ ! -d "${PWD}/results/bandwidth/p2p/tt" ]
      then
         mkdir "${PWD}/results/bandwidth/p2p/tt"
      fi
   fi
fi

if [ ! -d "${PWD}/results/contention" ]
then
   mkdir "${PWD}/results/contention"
   mkdir "${PWD}/results/contention/pcie"
   mkdir "${PWD}/results/contention/qpi"
   mkdir "${PWD}/results/contention/mem"
else
   if [ ! -d "${PWD}/results/contention/pcie" ]
   then
      mkdir "${PWD}/results/contention/pcie"
   fi

   if [ ! -d "${PWD}/results/contention/qpi" ]
   then
      mkdir "${PWD}/results/contention/qpi"
   fi

   if [ ! -d "${PWD}/results/contention/mem" ]
   then
      mkdir "${PWD}/results/contention/mem"
   fi
fi

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
echo -e "\n\e[34m\e[1mMapping system topology... ${PWD}/results/\e[0m\e[1m\e[31m$topo_hwloc\e[0m\e[34m\e[1m and ${PWD}/results/\e[0m\e[1m\e[31m$topo_scan\e[0m\n"

if [ -f "${PWD}/results/$topo_hwloc" ]
then
   rm ${PWD}/results/$topo_hwloc
fi

# custom topology scan script
# only works with bash/unix environment
${PWD}/scripts/map_topology.sh "${PWD}/results/$topo_scan"
cat ${PWD}/results/$topo_scan

echo ""

# hwloc command line topology tree stdout print
lstopo -v >> ${PWD}/results/$topo_hwloc
cat ${PWD}/results/$topo_hwloc

# check for user benchmark parameter file
if [[ $1 != '' ]]
then
   bench_params="$1"
   echo -e "\n\e[34m\e[1mInitiating benchmark (\e[1m\e[31m$exe\e[0m\e[34m\e[1m) with parameter file:\e[0m \e[1m\e[31m$bench_params\e[0m\n"
else
   echo -e "\n\e[34m\e[1mInitiating benchmark with default parameters:\e[0m\n"
fi

sleep 1

# run bandwidth benchmark
#eval ./$exe $bench_params

# plot data and save graphs
echo -e "\e[34m\e[1mRunning plotting scripts:\e[0m \e[1m\e[31m$plot_script\e[0m\n"
${PWD}/scripts/plot_data.sh


