#!/bin/bash


if [[ "${PWD}" == *"scripts"* ]]
then
   cd ..
fi

./scripts/clean_results.sh
rm ./results/*.out
rm ./results/topology.in
make clean
