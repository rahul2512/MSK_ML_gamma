#!/bin/bash -l
#SBATCH -J RNN_196
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7020 &
python  ${path}/main.py 7021 &
python  ${path}/main.py 7022 &
python  ${path}/main.py 7023 &
python  ${path}/main.py 7024 &
python  ${path}/main.py 7025 &
python  ${path}/main.py 7026 &
python  ${path}/main.py 7027 &
python  ${path}/main.py 7028 &
python  ${path}/main.py 7029 &
python  ${path}/main.py 7030 &
python  ${path}/main.py 7031 &
python  ${path}/main.py 7032 &
python  ${path}/main.py 7033 &
python  ${path}/main.py 7034 &
python  ${path}/main.py 7035 &
python  ${path}/main.py 7036 &
python  ${path}/main.py 7037 &
python  ${path}/main.py 7038 &
python  ${path}/main.py 7039 &
python  ${path}/main.py 7040 &
python  ${path}/main.py 7041 &
python  ${path}/main.py 7042 &
python  ${path}/main.py 7043 &
python  ${path}/main.py 7044 &
python  ${path}/main.py 7045 &
python  ${path}/main.py 7046 &
python  ${path}/main.py 7047 &
python  ${path}/main.py 7048 &
python  ${path}/main.py 7049 &
python  ${path}/main.py 7050 &
python  ${path}/main.py 7051 &
python  ${path}/main.py 7052 &
python  ${path}/main.py 7053 &
python  ${path}/main.py 7054 &
python  ${path}/main.py 7055 &
wait
