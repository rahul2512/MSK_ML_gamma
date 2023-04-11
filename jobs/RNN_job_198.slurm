#!/bin/bash -l
#SBATCH -J RNN_198
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7092 &
python  ${path}/main.py 7093 &
python  ${path}/main.py 7094 &
python  ${path}/main.py 7095 &
python  ${path}/main.py 7096 &
python  ${path}/main.py 7097 &
python  ${path}/main.py 7098 &
python  ${path}/main.py 7099 &
python  ${path}/main.py 7100 &
python  ${path}/main.py 7101 &
python  ${path}/main.py 7102 &
python  ${path}/main.py 7103 &
python  ${path}/main.py 7104 &
python  ${path}/main.py 7105 &
python  ${path}/main.py 7106 &
python  ${path}/main.py 7107 &
python  ${path}/main.py 7108 &
python  ${path}/main.py 7109 &
python  ${path}/main.py 7110 &
python  ${path}/main.py 7111 &
python  ${path}/main.py 7112 &
python  ${path}/main.py 7113 &
python  ${path}/main.py 7114 &
python  ${path}/main.py 7115 &
python  ${path}/main.py 7116 &
python  ${path}/main.py 7117 &
python  ${path}/main.py 7118 &
python  ${path}/main.py 7119 &
python  ${path}/main.py 7120 &
python  ${path}/main.py 7121 &
python  ${path}/main.py 7122 &
python  ${path}/main.py 7123 &
python  ${path}/main.py 7124 &
python  ${path}/main.py 7125 &
python  ${path}/main.py 7126 &
python  ${path}/main.py 7127 &
wait
