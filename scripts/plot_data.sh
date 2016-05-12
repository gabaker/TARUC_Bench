#!/bin/bash

curr_dir="$PWD"

#move to the main benchmark directory if in in scripts/
if [ $curr_dir == *"scripts"* ]
then
   cd .. 
fi

echo -e "\n\e[34m\e[1mPlotting Memory Management Overhead Micro-benchmark Data...\e[0m"
find ./ -name "*_overhead.csv"

echo -e "\n\e[34m\e[1mPlotting Host-Host Memory Transfer Micro-benchmark Data...\e[0m"
find ./ -name "*_ranged_hh_bw.csv"
find ./ -name "*_ranged_hh_tt.csv"

echo -e "\n\e[34m\e[1mPlotting Host-Device Memory Transfer Micro-benchmark Data...\e[0m"
find ./ -name "*_ranged_hd_bw.csv"
find ./ -name "*_ranged_hd_tt.csv"

echo -e "\n\e[34m\e[1mPlotting Device-Device Memory Transfer Micro-benchmark Data...\e[0m"
find ./ -name "*_ranged_p2p_bw.csv"
find ./ -name "*_ranged_p2p_tt.csv"

echo -e "\n\e[34m\e[1mPlotting Resource Contention Micro-benchmark Data...\e[0m"

