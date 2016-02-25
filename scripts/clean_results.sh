#!/bin/bash

if [[ "${PWD}" == *"scripts"* ]]
then
   cd ..
fi

rm ./results/*.png 
rm ./results/*.csv 

