#!/bin/bash -l
#SBATCH -J RNN_251
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9000 &
python  ${path}/main.py 9001 &
python  ${path}/main.py 9002 &
python  ${path}/main.py 9003 &
python  ${path}/main.py 9004 &
python  ${path}/main.py 9005 &
python  ${path}/main.py 9006 &
python  ${path}/main.py 9007 &
python  ${path}/main.py 9008 &
python  ${path}/main.py 9009 &
python  ${path}/main.py 9010 &
python  ${path}/main.py 9011 &
python  ${path}/main.py 9012 &
python  ${path}/main.py 9013 &
python  ${path}/main.py 9014 &
python  ${path}/main.py 9015 &
python  ${path}/main.py 9016 &
python  ${path}/main.py 9017 &
python  ${path}/main.py 9018 &
python  ${path}/main.py 9019 &
python  ${path}/main.py 9020 &
python  ${path}/main.py 9021 &
python  ${path}/main.py 9022 &
python  ${path}/main.py 9023 &
python  ${path}/main.py 9024 &
python  ${path}/main.py 9025 &
python  ${path}/main.py 9026 &
python  ${path}/main.py 9027 &
python  ${path}/main.py 9028 &
python  ${path}/main.py 9029 &
python  ${path}/main.py 9030 &
python  ${path}/main.py 9031 &
python  ${path}/main.py 9032 &
python  ${path}/main.py 9033 &
python  ${path}/main.py 9034 &
python  ${path}/main.py 9035 &
wait
