#!/bin/bash -l
#SBATCH -J RNN_286
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10260 &
python  ${path}/main.py 10261 &
python  ${path}/main.py 10262 &
python  ${path}/main.py 10263 &
python  ${path}/main.py 10264 &
python  ${path}/main.py 10265 &
python  ${path}/main.py 10266 &
python  ${path}/main.py 10267 &
python  ${path}/main.py 10268 &
python  ${path}/main.py 10269 &
python  ${path}/main.py 10270 &
python  ${path}/main.py 10271 &
python  ${path}/main.py 10272 &
python  ${path}/main.py 10273 &
python  ${path}/main.py 10274 &
python  ${path}/main.py 10275 &
python  ${path}/main.py 10276 &
python  ${path}/main.py 10277 &
python  ${path}/main.py 10278 &
python  ${path}/main.py 10279 &
python  ${path}/main.py 10280 &
python  ${path}/main.py 10281 &
python  ${path}/main.py 10282 &
python  ${path}/main.py 10283 &
python  ${path}/main.py 10284 &
python  ${path}/main.py 10285 &
python  ${path}/main.py 10286 &
python  ${path}/main.py 10287 &
python  ${path}/main.py 10288 &
python  ${path}/main.py 10289 &
python  ${path}/main.py 10290 &
python  ${path}/main.py 10291 &
python  ${path}/main.py 10292 &
python  ${path}/main.py 10293 &
python  ${path}/main.py 10294 &
python  ${path}/main.py 10295 &
wait
