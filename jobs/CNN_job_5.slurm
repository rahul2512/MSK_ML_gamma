#!/bin/bash -l
#SBATCH -J CNN_5
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 96 &
python  ${path}/main.py 97 &
python  ${path}/main.py 98 &
python  ${path}/main.py 99 &
python  ${path}/main.py 100 &
python  ${path}/main.py 101 &
python  ${path}/main.py 102 &
python  ${path}/main.py 103 &
python  ${path}/main.py 104 &
python  ${path}/main.py 105 &
python  ${path}/main.py 106 &
python  ${path}/main.py 107 &
python  ${path}/main.py 108 &
python  ${path}/main.py 109 &
python  ${path}/main.py 110 &
python  ${path}/main.py 111 &
python  ${path}/main.py 112 &
python  ${path}/main.py 113 &
python  ${path}/main.py 114 &
python  ${path}/main.py 115 &
python  ${path}/main.py 116 &
python  ${path}/main.py 117 &
python  ${path}/main.py 118 &
python  ${path}/main.py 119 &
wait
