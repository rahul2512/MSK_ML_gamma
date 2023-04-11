#!/bin/bash -l
#SBATCH -J RNN_252
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9036 &
python  ${path}/main.py 9037 &
python  ${path}/main.py 9038 &
python  ${path}/main.py 9039 &
python  ${path}/main.py 9040 &
python  ${path}/main.py 9041 &
python  ${path}/main.py 9042 &
python  ${path}/main.py 9043 &
python  ${path}/main.py 9044 &
python  ${path}/main.py 9045 &
python  ${path}/main.py 9046 &
python  ${path}/main.py 9047 &
python  ${path}/main.py 9048 &
python  ${path}/main.py 9049 &
python  ${path}/main.py 9050 &
python  ${path}/main.py 9051 &
python  ${path}/main.py 9052 &
python  ${path}/main.py 9053 &
python  ${path}/main.py 9054 &
python  ${path}/main.py 9055 &
python  ${path}/main.py 9056 &
python  ${path}/main.py 9057 &
python  ${path}/main.py 9058 &
python  ${path}/main.py 9059 &
python  ${path}/main.py 9060 &
python  ${path}/main.py 9061 &
python  ${path}/main.py 9062 &
python  ${path}/main.py 9063 &
python  ${path}/main.py 9064 &
python  ${path}/main.py 9065 &
python  ${path}/main.py 9066 &
python  ${path}/main.py 9067 &
python  ${path}/main.py 9068 &
python  ${path}/main.py 9069 &
python  ${path}/main.py 9070 &
python  ${path}/main.py 9071 &
wait
