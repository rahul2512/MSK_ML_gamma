#!/bin/bash -l
#SBATCH -J RNN_168
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6012 &
python  ${path}/main.py 6013 &
python  ${path}/main.py 6014 &
python  ${path}/main.py 6015 &
python  ${path}/main.py 6016 &
python  ${path}/main.py 6017 &
python  ${path}/main.py 6018 &
python  ${path}/main.py 6019 &
python  ${path}/main.py 6020 &
python  ${path}/main.py 6021 &
python  ${path}/main.py 6022 &
python  ${path}/main.py 6023 &
python  ${path}/main.py 6024 &
python  ${path}/main.py 6025 &
python  ${path}/main.py 6026 &
python  ${path}/main.py 6027 &
python  ${path}/main.py 6028 &
python  ${path}/main.py 6029 &
python  ${path}/main.py 6030 &
python  ${path}/main.py 6031 &
python  ${path}/main.py 6032 &
python  ${path}/main.py 6033 &
python  ${path}/main.py 6034 &
python  ${path}/main.py 6035 &
python  ${path}/main.py 6036 &
python  ${path}/main.py 6037 &
python  ${path}/main.py 6038 &
python  ${path}/main.py 6039 &
python  ${path}/main.py 6040 &
python  ${path}/main.py 6041 &
python  ${path}/main.py 6042 &
python  ${path}/main.py 6043 &
python  ${path}/main.py 6044 &
python  ${path}/main.py 6045 &
python  ${path}/main.py 6046 &
python  ${path}/main.py 6047 &
wait
