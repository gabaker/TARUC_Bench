#!/bin/bash

if [[ "${PWD}" == *"scripts"* ]]
then
   cd ..
fi

./scripts/clean_results.sh
make clean
