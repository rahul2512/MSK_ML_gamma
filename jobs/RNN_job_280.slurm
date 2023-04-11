#!/bin/bash -l
#SBATCH -J RNN_280
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10044 &
python  ${path}/main.py 10045 &
python  ${path}/main.py 10046 &
python  ${path}/main.py 10047 &
python  ${path}/main.py 10048 &
python  ${path}/main.py 10049 &
python  ${path}/main.py 10050 &
python  ${path}/main.py 10051 &
python  ${path}/main.py 10052 &
python  ${path}/main.py 10053 &
python  ${path}/main.py 10054 &
python  ${path}/main.py 10055 &
python  ${path}/main.py 10056 &
python  ${path}/main.py 10057 &
python  ${path}/main.py 10058 &
python  ${path}/main.py 10059 &
python  ${path}/main.py 10060 &
python  ${path}/main.py 10061 &
python  ${path}/main.py 10062 &
python  ${path}/main.py 10063 &
python  ${path}/main.py 10064 &
python  ${path}/main.py 10065 &
python  ${path}/main.py 10066 &
python  ${path}/main.py 10067 &
python  ${path}/main.py 10068 &
python  ${path}/main.py 10069 &
python  ${path}/main.py 10070 &
python  ${path}/main.py 10071 &
python  ${path}/main.py 10072 &
python  ${path}/main.py 10073 &
python  ${path}/main.py 10074 &
python  ${path}/main.py 10075 &
python  ${path}/main.py 10076 &
python  ${path}/main.py 10077 &
python  ${path}/main.py 10078 &
python  ${path}/main.py 10079 &
wait
