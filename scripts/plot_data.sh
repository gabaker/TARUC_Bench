#!/bin/bash

curr_dir="$PWD"
#move to the main benchmark directory if in in scripts/
if [[ $curr_dir == *"scripts"* || $curr_dir == *"results"* ]]
then
   cd ..
   curr_dir="$PWD"
fi

main_dir="$PWD"
script_dir="$PWD/scripts"

if [[ ! $curr_dir == *"results"* ]];
then
   cd $main_dir/results
   curr_dir="$PWD"
fi

#-----------------------------------------------------------------------------
if [[ $(find ./ -name "*_overhead.csv") ]]
then
   echo -e "\n\e[34m\e[1mPlotting Memory Management Overhead Micro-Benchmark Data...\e[0m"
   python2 $script_dir/plot_overhead.py $(find ./ -name "*_overhead.csv") 
fi 

#-----------------------------------------------------------------------------
if [[ $(find ./ -name "*ranged_hh*") ]]
then
   echo -e "\n\e[34m\e[1mPlotting Host-Host Memory Transfer Micro-Benchmark Data...\e[0m"
fi

if [[ $(find ./ -name "*ranged_hh_bw.csv") ]]
then
   python2 $script_dir/plot_bandwidth_hh.py $(find ./ -name "*_ranged_hh_bw.csv") 
fi 

if [[ $(find ./ -name "*ranged_hh_tt.csv") ]]
then
   python2 $script_dir/plot_bandwidth_hh.py $(find ./ -name "*_ranged_hh_tt.csv") 
fi 

#-----------------------------------------------------------------------------

if [[ $(find ./ -name "*ranged_hd*") ]]
then
   echo -e "\n\e[34m\e[1mPlotting Host-Device Memory Transfer Micro-Benchmark Data...\e[0m"

   if [[ $(find ./ -name "*ranged_hd_bw.csv") ]]
   then
      python2 $script_dir/plot_bandwidth_hd.py $(find ./ -name "*_ranged_hd_bw.csv") 
   fi 

   if [[ $(find ./ -name "*ranged_hd_tt.csv") ]]
   then
      python2 $script_dir/plot_bandwidth_hd.py $(find ./ -name "*_ranged_hd_tt.csv") 
   fi 

fi

#-----------------------------------------------------------------------------
if [[ $(find ./ -name "*ranged_p2p*") ]]
then
   echo -e "\n\e[34m\e[1mPlotting Device-Device Memory Transfer Micro-Benchmark Data...\e[0m"

   if [[ $(find ./ -name "*ranged_p2p_bw.csv") ]]
   then
      python2 $script_dir/plot_bandwidth_p2p.py $(find ./ -name "*_ranged_p2p_bw.csv") 
   fi 

   if [[ $(find ./ -name "*ranged_p2p_tt.csv") ]]
   then
      python2 $script_dir/plot_bandwidth_p2p.py $(find ./ -name "*_ranged_p2p_tt.csv") 
   fi 
fi


#-----------------------------------------------------------------------------
if [[ $(find ./ -name "*_random_access.csv") ]]
then
   echo -e "\n\e[34m\e[1mPlotting Non-Uniform Random Memory Access (NURMA) Micro-Benchmark Data...\e[0m"

   python2 $script_dir/plot_random_access.py $(find ./ -name "*_random_access.csv") 
fi

#-----------------------------------------------------------------------------
if [[ $(find ./ -name "*_mem_contention.csv") ]]
then
   echo -e "\n\e[34m\e[1mPlotting Local Memory Resource Contention Micro-Benchmark Data...\e[0m"

   python2 $script_dir/plot_mem_contention.py $(find ./ -name "*_mem_contention.csv") 
fi

#-----------------------------------------------------------------------------
if [[ $(find ./ -name "*_qpi_contention.csv") ]]
then
   echo -e "\n\e[34m\e[1mPlotting Inter-Socket Communication Contention Micro-Benchmark Data...\e[0m"

   python2 $script_dir/plot_qpi_contention.py $(find ./ -name "*_qpi_contention.csv") 
fi

#-----------------------------------------------------------------------------
if [[ $(find ./ -name "*_pcie_contention.csv") ]]
then
   echo -e "\n\e[34m\e[1mPlotting Host-Device Resource Usage and Contention Micro-Benchmark Data...\e[0m"

   python2 $script_dir/plot_pcie_contention.py $(find ./ -name "*_pcie_contention.csv") 
fi


