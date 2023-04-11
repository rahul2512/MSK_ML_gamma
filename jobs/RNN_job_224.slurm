#!/bin/bash -l
#SBATCH -J RNN_224
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8028 &
python  ${path}/main.py 8029 &
python  ${path}/main.py 8030 &
python  ${path}/main.py 8031 &
python  ${path}/main.py 8032 &
python  ${path}/main.py 8033 &
python  ${path}/main.py 8034 &
python  ${path}/main.py 8035 &
python  ${path}/main.py 8036 &
python  ${path}/main.py 8037 &
python  ${path}/main.py 8038 &
python  ${path}/main.py 8039 &
python  ${path}/main.py 8040 &
python  ${path}/main.py 8041 &
python  ${path}/main.py 8042 &
python  ${path}/main.py 8043 &
python  ${path}/main.py 8044 &
python  ${path}/main.py 8045 &
python  ${path}/main.py 8046 &
python  ${path}/main.py 8047 &
python  ${path}/main.py 8048 &
python  ${path}/main.py 8049 &
python  ${path}/main.py 8050 &
python  ${path}/main.py 8051 &
python  ${path}/main.py 8052 &
python  ${path}/main.py 8053 &
python  ${path}/main.py 8054 &
python  ${path}/main.py 8055 &
python  ${path}/main.py 8056 &
python  ${path}/main.py 8057 &
python  ${path}/main.py 8058 &
python  ${path}/main.py 8059 &
python  ${path}/main.py 8060 &
python  ${path}/main.py 8061 &
python  ${path}/main.py 8062 &
python  ${path}/main.py 8063 &
wait
