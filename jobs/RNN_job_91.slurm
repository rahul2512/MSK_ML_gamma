#!/bin/bash -l
#SBATCH -J RNN_91
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3240 &
python  ${path}/main.py 3241 &
python  ${path}/main.py 3242 &
python  ${path}/main.py 3243 &
python  ${path}/main.py 3244 &
python  ${path}/main.py 3245 &
python  ${path}/main.py 3246 &
python  ${path}/main.py 3247 &
python  ${path}/main.py 3248 &
python  ${path}/main.py 3249 &
python  ${path}/main.py 3250 &
python  ${path}/main.py 3251 &
python  ${path}/main.py 3252 &
python  ${path}/main.py 3253 &
python  ${path}/main.py 3254 &
python  ${path}/main.py 3255 &
python  ${path}/main.py 3256 &
python  ${path}/main.py 3257 &
python  ${path}/main.py 3258 &
python  ${path}/main.py 3259 &
python  ${path}/main.py 3260 &
python  ${path}/main.py 3261 &
python  ${path}/main.py 3262 &
python  ${path}/main.py 3263 &
python  ${path}/main.py 3264 &
python  ${path}/main.py 3265 &
python  ${path}/main.py 3266 &
python  ${path}/main.py 3267 &
python  ${path}/main.py 3268 &
python  ${path}/main.py 3269 &
python  ${path}/main.py 3270 &
python  ${path}/main.py 3271 &
python  ${path}/main.py 3272 &
python  ${path}/main.py 3273 &
python  ${path}/main.py 3274 &
python  ${path}/main.py 3275 &
wait
