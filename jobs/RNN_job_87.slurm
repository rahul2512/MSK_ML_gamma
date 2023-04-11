#!/bin/bash -l
#SBATCH -J RNN_87
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3096 &
python  ${path}/main.py 3097 &
python  ${path}/main.py 3098 &
python  ${path}/main.py 3099 &
python  ${path}/main.py 3100 &
python  ${path}/main.py 3101 &
python  ${path}/main.py 3102 &
python  ${path}/main.py 3103 &
python  ${path}/main.py 3104 &
python  ${path}/main.py 3105 &
python  ${path}/main.py 3106 &
python  ${path}/main.py 3107 &
python  ${path}/main.py 3108 &
python  ${path}/main.py 3109 &
python  ${path}/main.py 3110 &
python  ${path}/main.py 3111 &
python  ${path}/main.py 3112 &
python  ${path}/main.py 3113 &
python  ${path}/main.py 3114 &
python  ${path}/main.py 3115 &
python  ${path}/main.py 3116 &
python  ${path}/main.py 3117 &
python  ${path}/main.py 3118 &
python  ${path}/main.py 3119 &
python  ${path}/main.py 3120 &
python  ${path}/main.py 3121 &
python  ${path}/main.py 3122 &
python  ${path}/main.py 3123 &
python  ${path}/main.py 3124 &
python  ${path}/main.py 3125 &
python  ${path}/main.py 3126 &
python  ${path}/main.py 3127 &
python  ${path}/main.py 3128 &
python  ${path}/main.py 3129 &
python  ${path}/main.py 3130 &
python  ${path}/main.py 3131 &
wait
