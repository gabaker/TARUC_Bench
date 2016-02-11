#!/bin/bash


if [[ "${PWD}" == *"scripts"* ]]
then
   cd ..
fi

./scripts/clean_results.sh
rm *.out
rm topology.in
make clean
