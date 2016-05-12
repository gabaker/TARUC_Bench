#!/bin/bash

if [[ "${PWD}" == *"scripts"* ]]
then
   cd ..
fi

rm -rf ./results/*.png 
rm -rf ./results/*.csv 
rm -rf ./results/*.out
rm -rf ./results/*.in
