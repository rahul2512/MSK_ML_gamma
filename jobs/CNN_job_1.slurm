#!/bin/bash -l
#SBATCH -J CNN_1
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 0 &
python  ${path}/main.py 1 &
python  ${path}/main.py 2 &
python  ${path}/main.py 3 &
python  ${path}/main.py 4 &
python  ${path}/main.py 5 &
python  ${path}/main.py 6 &
python  ${path}/main.py 7 &
python  ${path}/main.py 8 &
python  ${path}/main.py 9 &
python  ${path}/main.py 10 &
python  ${path}/main.py 11 &
python  ${path}/main.py 12 &
python  ${path}/main.py 13 &
python  ${path}/main.py 14 &
python  ${path}/main.py 15 &
python  ${path}/main.py 16 &
python  ${path}/main.py 17 &
python  ${path}/main.py 18 &
python  ${path}/main.py 19 &
python  ${path}/main.py 20 &
python  ${path}/main.py 21 &
python  ${path}/main.py 22 &
python  ${path}/main.py 23 &
wait
