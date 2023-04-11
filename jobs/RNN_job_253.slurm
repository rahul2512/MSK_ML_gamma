#!/bin/bash -l
#SBATCH -J RNN_253
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9072 &
python  ${path}/main.py 9073 &
python  ${path}/main.py 9074 &
python  ${path}/main.py 9075 &
python  ${path}/main.py 9076 &
python  ${path}/main.py 9077 &
python  ${path}/main.py 9078 &
python  ${path}/main.py 9079 &
python  ${path}/main.py 9080 &
python  ${path}/main.py 9081 &
python  ${path}/main.py 9082 &
python  ${path}/main.py 9083 &
python  ${path}/main.py 9084 &
python  ${path}/main.py 9085 &
python  ${path}/main.py 9086 &
python  ${path}/main.py 9087 &
python  ${path}/main.py 9088 &
python  ${path}/main.py 9089 &
python  ${path}/main.py 9090 &
python  ${path}/main.py 9091 &
python  ${path}/main.py 9092 &
python  ${path}/main.py 9093 &
python  ${path}/main.py 9094 &
python  ${path}/main.py 9095 &
python  ${path}/main.py 9096 &
python  ${path}/main.py 9097 &
python  ${path}/main.py 9098 &
python  ${path}/main.py 9099 &
python  ${path}/main.py 9100 &
python  ${path}/main.py 9101 &
python  ${path}/main.py 9102 &
python  ${path}/main.py 9103 &
python  ${path}/main.py 9104 &
python  ${path}/main.py 9105 &
python  ${path}/main.py 9106 &
python  ${path}/main.py 9107 &
wait
