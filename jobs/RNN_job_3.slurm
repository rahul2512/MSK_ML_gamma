#!/bin/bash -l
#SBATCH -J RNN_3
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 72 &
python  ${path}/main.py 73 &
python  ${path}/main.py 74 &
python  ${path}/main.py 75 &
python  ${path}/main.py 76 &
python  ${path}/main.py 77 &
python  ${path}/main.py 78 &
python  ${path}/main.py 79 &
python  ${path}/main.py 80 &
python  ${path}/main.py 81 &
python  ${path}/main.py 82 &
python  ${path}/main.py 83 &
python  ${path}/main.py 84 &
python  ${path}/main.py 85 &
python  ${path}/main.py 86 &
python  ${path}/main.py 87 &
python  ${path}/main.py 88 &
python  ${path}/main.py 89 &
python  ${path}/main.py 90 &
python  ${path}/main.py 91 &
python  ${path}/main.py 92 &
python  ${path}/main.py 93 &
python  ${path}/main.py 94 &
python  ${path}/main.py 95 &
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
wait
