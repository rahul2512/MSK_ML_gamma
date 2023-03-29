#!/bin/bash -l
#SBATCH -J CNN_2
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 24 &
python  ${path}/main.py 25 &
python  ${path}/main.py 26 &
python  ${path}/main.py 27 &
python  ${path}/main.py 28 &
python  ${path}/main.py 29 &
python  ${path}/main.py 30 &
python  ${path}/main.py 31 &
python  ${path}/main.py 32 &
python  ${path}/main.py 33 &
python  ${path}/main.py 34 &
python  ${path}/main.py 35 &
python  ${path}/main.py 36 &
python  ${path}/main.py 37 &
python  ${path}/main.py 38 &
python  ${path}/main.py 39 &
python  ${path}/main.py 40 &
python  ${path}/main.py 41 &
python  ${path}/main.py 42 &
python  ${path}/main.py 43 &
python  ${path}/main.py 44 &
python  ${path}/main.py 45 &
python  ${path}/main.py 46 &
python  ${path}/main.py 47 &
wait
