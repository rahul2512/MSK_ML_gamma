#!/bin/bash -l
#SBATCH -J RNN_178
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6372 &
python  ${path}/main.py 6373 &
python  ${path}/main.py 6374 &
python  ${path}/main.py 6375 &
python  ${path}/main.py 6376 &
python  ${path}/main.py 6377 &
python  ${path}/main.py 6378 &
python  ${path}/main.py 6379 &
python  ${path}/main.py 6380 &
python  ${path}/main.py 6381 &
python  ${path}/main.py 6382 &
python  ${path}/main.py 6383 &
python  ${path}/main.py 6384 &
python  ${path}/main.py 6385 &
python  ${path}/main.py 6386 &
python  ${path}/main.py 6387 &
python  ${path}/main.py 6388 &
python  ${path}/main.py 6389 &
python  ${path}/main.py 6390 &
python  ${path}/main.py 6391 &
python  ${path}/main.py 6392 &
python  ${path}/main.py 6393 &
python  ${path}/main.py 6394 &
python  ${path}/main.py 6395 &
python  ${path}/main.py 6396 &
python  ${path}/main.py 6397 &
python  ${path}/main.py 6398 &
python  ${path}/main.py 6399 &
python  ${path}/main.py 6400 &
python  ${path}/main.py 6401 &
python  ${path}/main.py 6402 &
python  ${path}/main.py 6403 &
python  ${path}/main.py 6404 &
python  ${path}/main.py 6405 &
python  ${path}/main.py 6406 &
python  ${path}/main.py 6407 &
wait
