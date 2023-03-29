#!/bin/bash -l
#SBATCH -J CNN_3
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
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
