#!/bin/bash -l
#SBATCH -J RNN_88
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3132 &
python  ${path}/main.py 3133 &
python  ${path}/main.py 3134 &
python  ${path}/main.py 3135 &
python  ${path}/main.py 3136 &
python  ${path}/main.py 3137 &
python  ${path}/main.py 3138 &
python  ${path}/main.py 3139 &
python  ${path}/main.py 3140 &
python  ${path}/main.py 3141 &
python  ${path}/main.py 3142 &
python  ${path}/main.py 3143 &
python  ${path}/main.py 3144 &
python  ${path}/main.py 3145 &
python  ${path}/main.py 3146 &
python  ${path}/main.py 3147 &
python  ${path}/main.py 3148 &
python  ${path}/main.py 3149 &
python  ${path}/main.py 3150 &
python  ${path}/main.py 3151 &
python  ${path}/main.py 3152 &
python  ${path}/main.py 3153 &
python  ${path}/main.py 3154 &
python  ${path}/main.py 3155 &
python  ${path}/main.py 3156 &
python  ${path}/main.py 3157 &
python  ${path}/main.py 3158 &
python  ${path}/main.py 3159 &
python  ${path}/main.py 3160 &
python  ${path}/main.py 3161 &
python  ${path}/main.py 3162 &
python  ${path}/main.py 3163 &
python  ${path}/main.py 3164 &
python  ${path}/main.py 3165 &
python  ${path}/main.py 3166 &
python  ${path}/main.py 3167 &
wait
