#!/bin/bash -l
#SBATCH -J CNNLSTM_1
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
python  ${path}/main.py 48 &
python  ${path}/main.py 49 &
python  ${path}/main.py 50 &
python  ${path}/main.py 51 &
python  ${path}/main.py 52 &
python  ${path}/main.py 53 &
python  ${path}/main.py 54 &
python  ${path}/main.py 55 &
python  ${path}/main.py 56 &
python  ${path}/main.py 57 &
python  ${path}/main.py 58 &
python  ${path}/main.py 59 &
python  ${path}/main.py 60 &
python  ${path}/main.py 61 &
python  ${path}/main.py 62 &
python  ${path}/main.py 63 &
python  ${path}/main.py 64 &
python  ${path}/main.py 65 &
python  ${path}/main.py 66 &
python  ${path}/main.py 67 &
python  ${path}/main.py 68 &
python  ${path}/main.py 69 &
python  ${path}/main.py 70 &
python  ${path}/main.py 71 &
wait
