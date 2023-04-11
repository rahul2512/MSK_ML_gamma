#!/bin/bash -l
#SBATCH -J RNN_170
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6084 &
python  ${path}/main.py 6085 &
python  ${path}/main.py 6086 &
python  ${path}/main.py 6087 &
python  ${path}/main.py 6088 &
python  ${path}/main.py 6089 &
python  ${path}/main.py 6090 &
python  ${path}/main.py 6091 &
python  ${path}/main.py 6092 &
python  ${path}/main.py 6093 &
python  ${path}/main.py 6094 &
python  ${path}/main.py 6095 &
python  ${path}/main.py 6096 &
python  ${path}/main.py 6097 &
python  ${path}/main.py 6098 &
python  ${path}/main.py 6099 &
python  ${path}/main.py 6100 &
python  ${path}/main.py 6101 &
python  ${path}/main.py 6102 &
python  ${path}/main.py 6103 &
python  ${path}/main.py 6104 &
python  ${path}/main.py 6105 &
python  ${path}/main.py 6106 &
python  ${path}/main.py 6107 &
python  ${path}/main.py 6108 &
python  ${path}/main.py 6109 &
python  ${path}/main.py 6110 &
python  ${path}/main.py 6111 &
python  ${path}/main.py 6112 &
python  ${path}/main.py 6113 &
python  ${path}/main.py 6114 &
python  ${path}/main.py 6115 &
python  ${path}/main.py 6116 &
python  ${path}/main.py 6117 &
python  ${path}/main.py 6118 &
python  ${path}/main.py 6119 &
wait
