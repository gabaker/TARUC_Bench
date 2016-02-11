#!/bin/bash

if [[ "${PWD}" == *"scripts"* ]]
then
   cd ..
fi

rm *.png 
rm *.csv 
rm benchmark_params.out device_info.out
rm topology.in
