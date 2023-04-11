#!/bin/bash -l
#SBATCH -J RNN_177
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6336 &
python  ${path}/main.py 6337 &
python  ${path}/main.py 6338 &
python  ${path}/main.py 6339 &
python  ${path}/main.py 6340 &
python  ${path}/main.py 6341 &
python  ${path}/main.py 6342 &
python  ${path}/main.py 6343 &
python  ${path}/main.py 6344 &
python  ${path}/main.py 6345 &
python  ${path}/main.py 6346 &
python  ${path}/main.py 6347 &
python  ${path}/main.py 6348 &
python  ${path}/main.py 6349 &
python  ${path}/main.py 6350 &
python  ${path}/main.py 6351 &
python  ${path}/main.py 6352 &
python  ${path}/main.py 6353 &
python  ${path}/main.py 6354 &
python  ${path}/main.py 6355 &
python  ${path}/main.py 6356 &
python  ${path}/main.py 6357 &
python  ${path}/main.py 6358 &
python  ${path}/main.py 6359 &
python  ${path}/main.py 6360 &
python  ${path}/main.py 6361 &
python  ${path}/main.py 6362 &
python  ${path}/main.py 6363 &
python  ${path}/main.py 6364 &
python  ${path}/main.py 6365 &
python  ${path}/main.py 6366 &
python  ${path}/main.py 6367 &
python  ${path}/main.py 6368 &
python  ${path}/main.py 6369 &
python  ${path}/main.py 6370 &
python  ${path}/main.py 6371 &
wait
