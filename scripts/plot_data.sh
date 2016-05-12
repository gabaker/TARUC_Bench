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

echo -e "\n\e[34m\e[1mPlotting Memory Management Overhead Micro-benchmark Data...\e[0m"
if [[ $(find ./ -name "*_overhead.csv") ]]
then
   python2 $script_dir/plot_overhead.py $(find ./ -name "*_overhead.csv") 
fi 

echo -e "\n\e[34m\e[1mPlotting Host-Host Memory Transfer Micro-benchmark Data...\e[0m"
if [[ $(find ./ -name "*ranged_hh_bw.csv") ]]
then
   python2 $script_dir/plot_bandwidth_hh.py $(find ./ -name "*_ranged_hh_bw.csv") 
fi 

if [[ $(find ./ -name "*ranged_hh_tt.csv") ]]
then
   python2 $script_dir/plot_bandwidth_hh.py $(find ./ -name "*_ranged_hh_tt.csv") 
fi 

echo -e "\n\e[34m\e[1mPlotting Host-Device Memory Transfer Micro-benchmark Data...\e[0m"
if [[ $(find ./ -name "*ranged_hd_bw.csv") ]]
then
   python2 $script_dir/plot_bandwidth_hd.py $(find ./ -name "*_ranged_hd_bw.csv") 
fi 

if [[ $(find ./ -name "*ranged_hd_tt.csv") ]]
then
   python2 $script_dir/plot_bandwidth_hd.py $(find ./ -name "*_ranged_hd_tt.csv") 
fi 

echo -e "\n\e[34m\e[1mPlotting Device-Device Memory Transfer Micro-benchmark Data...\e[0m"
if [[ $(find ./ -name "*ranged_p2p_bw.csv") ]]
then
   python2 $script_dir/plot_bandwidth_p2p.py $(find ./ -name "*_ranged_p2p_bw.csv") 
fi 

if [[ $(find ./ -name "*ranged_p2p_tt.csv") ]]
then
   python2 $script_dir/plot_bandwidth_p2p.py $(find ./ -name "*_ranged_p2p_tt.csv") 
fi 

echo -e "\n\e[34m\e[1mPlotting Resource Contention Micro-benchmark Data...\e[0m"
if [[ $(find ./ -name "*_contention.csv") ]]
then
   python2 $script_dir/plot_bandwidth.py $(find ./ -name "*_ranged_p2p_tt.csv") 
fi 


