#!/bin/bash -l
#SBATCH -J RNN_278
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9972 &
python  ${path}/main.py 9973 &
python  ${path}/main.py 9974 &
python  ${path}/main.py 9975 &
python  ${path}/main.py 9976 &
python  ${path}/main.py 9977 &
python  ${path}/main.py 9978 &
python  ${path}/main.py 9979 &
python  ${path}/main.py 9980 &
python  ${path}/main.py 9981 &
python  ${path}/main.py 9982 &
python  ${path}/main.py 9983 &
python  ${path}/main.py 9984 &
python  ${path}/main.py 9985 &
python  ${path}/main.py 9986 &
python  ${path}/main.py 9987 &
python  ${path}/main.py 9988 &
python  ${path}/main.py 9989 &
python  ${path}/main.py 9990 &
python  ${path}/main.py 9991 &
python  ${path}/main.py 9992 &
python  ${path}/main.py 9993 &
python  ${path}/main.py 9994 &
python  ${path}/main.py 9995 &
python  ${path}/main.py 9996 &
python  ${path}/main.py 9997 &
python  ${path}/main.py 9998 &
python  ${path}/main.py 9999 &
python  ${path}/main.py 10000 &
python  ${path}/main.py 10001 &
python  ${path}/main.py 10002 &
python  ${path}/main.py 10003 &
python  ${path}/main.py 10004 &
python  ${path}/main.py 10005 &
python  ${path}/main.py 10006 &
python  ${path}/main.py 10007 &
wait
