#!/usr/bin/env bash
set -o errexit # Exit on error
set -o nounset # Trigger error when expanding unset variables
export pvpython=./resources/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/bin/pvpython
cd .. # Move to the root directory
dataset=("vorts")
nu=1
lightType=Headlight
for i in ${dataset[@]};
do
  cur_data=$i
  echo "Rendering $cur_data"
  $pvpython ./apps/volTFRenderDataset.py ${cur_data} ./${cur_data}RGBa --l ${lightType} --c 'RGBA'  --nu ${nu} 
  $pvpython ./apps/volTFRenderDataset.py ${cur_data} ./${cur_data}RGBa --l ${lightType} --c 'RGBA'  --nu ${nu} -m test
done
  