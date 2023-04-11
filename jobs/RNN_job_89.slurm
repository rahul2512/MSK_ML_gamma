#!/bin/bash -l
#SBATCH -J RNN_89
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3168 &
python  ${path}/main.py 3169 &
python  ${path}/main.py 3170 &
python  ${path}/main.py 3171 &
python  ${path}/main.py 3172 &
python  ${path}/main.py 3173 &
python  ${path}/main.py 3174 &
python  ${path}/main.py 3175 &
python  ${path}/main.py 3176 &
python  ${path}/main.py 3177 &
python  ${path}/main.py 3178 &
python  ${path}/main.py 3179 &
python  ${path}/main.py 3180 &
python  ${path}/main.py 3181 &
python  ${path}/main.py 3182 &
python  ${path}/main.py 3183 &
python  ${path}/main.py 3184 &
python  ${path}/main.py 3185 &
python  ${path}/main.py 3186 &
python  ${path}/main.py 3187 &
python  ${path}/main.py 3188 &
python  ${path}/main.py 3189 &
python  ${path}/main.py 3190 &
python  ${path}/main.py 3191 &
python  ${path}/main.py 3192 &
python  ${path}/main.py 3193 &
python  ${path}/main.py 3194 &
python  ${path}/main.py 3195 &
python  ${path}/main.py 3196 &
python  ${path}/main.py 3197 &
python  ${path}/main.py 3198 &
python  ${path}/main.py 3199 &
python  ${path}/main.py 3200 &
python  ${path}/main.py 3201 &
python  ${path}/main.py 3202 &
python  ${path}/main.py 3203 &
wait
