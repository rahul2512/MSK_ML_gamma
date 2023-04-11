#!/bin/bash -l
#SBATCH -J RNN_250
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8964 &
python  ${path}/main.py 8965 &
python  ${path}/main.py 8966 &
python  ${path}/main.py 8967 &
python  ${path}/main.py 8968 &
python  ${path}/main.py 8969 &
python  ${path}/main.py 8970 &
python  ${path}/main.py 8971 &
python  ${path}/main.py 8972 &
python  ${path}/main.py 8973 &
python  ${path}/main.py 8974 &
python  ${path}/main.py 8975 &
python  ${path}/main.py 8976 &
python  ${path}/main.py 8977 &
python  ${path}/main.py 8978 &
python  ${path}/main.py 8979 &
python  ${path}/main.py 8980 &
python  ${path}/main.py 8981 &
python  ${path}/main.py 8982 &
python  ${path}/main.py 8983 &
python  ${path}/main.py 8984 &
python  ${path}/main.py 8985 &
python  ${path}/main.py 8986 &
python  ${path}/main.py 8987 &
python  ${path}/main.py 8988 &
python  ${path}/main.py 8989 &
python  ${path}/main.py 8990 &
python  ${path}/main.py 8991 &
python  ${path}/main.py 8992 &
python  ${path}/main.py 8993 &
python  ${path}/main.py 8994 &
python  ${path}/main.py 8995 &
python  ${path}/main.py 8996 &
python  ${path}/main.py 8997 &
python  ${path}/main.py 8998 &
python  ${path}/main.py 8999 &
wait
