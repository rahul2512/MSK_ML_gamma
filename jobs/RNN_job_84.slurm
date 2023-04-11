#!/bin/bash -l
#SBATCH -J RNN_84
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2988 &
python  ${path}/main.py 2989 &
python  ${path}/main.py 2990 &
python  ${path}/main.py 2991 &
python  ${path}/main.py 2992 &
python  ${path}/main.py 2993 &
python  ${path}/main.py 2994 &
python  ${path}/main.py 2995 &
python  ${path}/main.py 2996 &
python  ${path}/main.py 2997 &
python  ${path}/main.py 2998 &
python  ${path}/main.py 2999 &
python  ${path}/main.py 3000 &
python  ${path}/main.py 3001 &
python  ${path}/main.py 3002 &
python  ${path}/main.py 3003 &
python  ${path}/main.py 3004 &
python  ${path}/main.py 3005 &
python  ${path}/main.py 3006 &
python  ${path}/main.py 3007 &
python  ${path}/main.py 3008 &
python  ${path}/main.py 3009 &
python  ${path}/main.py 3010 &
python  ${path}/main.py 3011 &
python  ${path}/main.py 3012 &
python  ${path}/main.py 3013 &
python  ${path}/main.py 3014 &
python  ${path}/main.py 3015 &
python  ${path}/main.py 3016 &
python  ${path}/main.py 3017 &
python  ${path}/main.py 3018 &
python  ${path}/main.py 3019 &
python  ${path}/main.py 3020 &
python  ${path}/main.py 3021 &
python  ${path}/main.py 3022 &
python  ${path}/main.py 3023 &
wait
