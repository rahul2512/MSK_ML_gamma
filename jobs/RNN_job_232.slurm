#!/bin/bash -l
#SBATCH -J RNN_232
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8316 &
python  ${path}/main.py 8317 &
python  ${path}/main.py 8318 &
python  ${path}/main.py 8319 &
python  ${path}/main.py 8320 &
python  ${path}/main.py 8321 &
python  ${path}/main.py 8322 &
python  ${path}/main.py 8323 &
python  ${path}/main.py 8324 &
python  ${path}/main.py 8325 &
python  ${path}/main.py 8326 &
python  ${path}/main.py 8327 &
python  ${path}/main.py 8328 &
python  ${path}/main.py 8329 &
python  ${path}/main.py 8330 &
python  ${path}/main.py 8331 &
python  ${path}/main.py 8332 &
python  ${path}/main.py 8333 &
python  ${path}/main.py 8334 &
python  ${path}/main.py 8335 &
python  ${path}/main.py 8336 &
python  ${path}/main.py 8337 &
python  ${path}/main.py 8338 &
python  ${path}/main.py 8339 &
python  ${path}/main.py 8340 &
python  ${path}/main.py 8341 &
python  ${path}/main.py 8342 &
python  ${path}/main.py 8343 &
python  ${path}/main.py 8344 &
python  ${path}/main.py 8345 &
python  ${path}/main.py 8346 &
python  ${path}/main.py 8347 &
python  ${path}/main.py 8348 &
python  ${path}/main.py 8349 &
python  ${path}/main.py 8350 &
python  ${path}/main.py 8351 &
wait
