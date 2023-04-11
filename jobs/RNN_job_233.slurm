#!/bin/bash -l
#SBATCH -J RNN_233
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8352 &
python  ${path}/main.py 8353 &
python  ${path}/main.py 8354 &
python  ${path}/main.py 8355 &
python  ${path}/main.py 8356 &
python  ${path}/main.py 8357 &
python  ${path}/main.py 8358 &
python  ${path}/main.py 8359 &
python  ${path}/main.py 8360 &
python  ${path}/main.py 8361 &
python  ${path}/main.py 8362 &
python  ${path}/main.py 8363 &
python  ${path}/main.py 8364 &
python  ${path}/main.py 8365 &
python  ${path}/main.py 8366 &
python  ${path}/main.py 8367 &
python  ${path}/main.py 8368 &
python  ${path}/main.py 8369 &
python  ${path}/main.py 8370 &
python  ${path}/main.py 8371 &
python  ${path}/main.py 8372 &
python  ${path}/main.py 8373 &
python  ${path}/main.py 8374 &
python  ${path}/main.py 8375 &
python  ${path}/main.py 8376 &
python  ${path}/main.py 8377 &
python  ${path}/main.py 8378 &
python  ${path}/main.py 8379 &
python  ${path}/main.py 8380 &
python  ${path}/main.py 8381 &
python  ${path}/main.py 8382 &
python  ${path}/main.py 8383 &
python  ${path}/main.py 8384 &
python  ${path}/main.py 8385 &
python  ${path}/main.py 8386 &
python  ${path}/main.py 8387 &
wait
