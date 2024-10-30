#!/usr/bin/env bash
set -o errexit # Exit on error
set -o nounset # Trigger error when expanding unset variables
# define pvpython 
export pvpython=./resources/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/bin/pvpython
cd .. # Move to the root directory
dataset=("vorts")
nu=1
lightType=Orbital
TF_idx=10
for i in ${dataset[@]};
do
  cur_data=$i
  echo "Rendering $cur_data"
  $pvpython ./apps/volTFRelightGT.py ${cur_data} ./Relight${cur_data}RGB --l ${lightType} --c 'RGB'  --nu ${nu} -m test --TF ${TF_idx}
done
  