#!/bin/bash -l
#SBATCH -J RNN_167
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5976 &
python  ${path}/main.py 5977 &
python  ${path}/main.py 5978 &
python  ${path}/main.py 5979 &
python  ${path}/main.py 5980 &
python  ${path}/main.py 5981 &
python  ${path}/main.py 5982 &
python  ${path}/main.py 5983 &
python  ${path}/main.py 5984 &
python  ${path}/main.py 5985 &
python  ${path}/main.py 5986 &
python  ${path}/main.py 5987 &
python  ${path}/main.py 5988 &
python  ${path}/main.py 5989 &
python  ${path}/main.py 5990 &
python  ${path}/main.py 5991 &
python  ${path}/main.py 5992 &
python  ${path}/main.py 5993 &
python  ${path}/main.py 5994 &
python  ${path}/main.py 5995 &
python  ${path}/main.py 5996 &
python  ${path}/main.py 5997 &
python  ${path}/main.py 5998 &
python  ${path}/main.py 5999 &
python  ${path}/main.py 6000 &
python  ${path}/main.py 6001 &
python  ${path}/main.py 6002 &
python  ${path}/main.py 6003 &
python  ${path}/main.py 6004 &
python  ${path}/main.py 6005 &
python  ${path}/main.py 6006 &
python  ${path}/main.py 6007 &
python  ${path}/main.py 6008 &
python  ${path}/main.py 6009 &
python  ${path}/main.py 6010 &
python  ${path}/main.py 6011 &
wait
