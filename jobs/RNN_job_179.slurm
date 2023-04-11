#!/bin/bash -l
#SBATCH -J RNN_179
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6408 &
python  ${path}/main.py 6409 &
python  ${path}/main.py 6410 &
python  ${path}/main.py 6411 &
python  ${path}/main.py 6412 &
python  ${path}/main.py 6413 &
python  ${path}/main.py 6414 &
python  ${path}/main.py 6415 &
python  ${path}/main.py 6416 &
python  ${path}/main.py 6417 &
python  ${path}/main.py 6418 &
python  ${path}/main.py 6419 &
python  ${path}/main.py 6420 &
python  ${path}/main.py 6421 &
python  ${path}/main.py 6422 &
python  ${path}/main.py 6423 &
python  ${path}/main.py 6424 &
python  ${path}/main.py 6425 &
python  ${path}/main.py 6426 &
python  ${path}/main.py 6427 &
python  ${path}/main.py 6428 &
python  ${path}/main.py 6429 &
python  ${path}/main.py 6430 &
python  ${path}/main.py 6431 &
python  ${path}/main.py 6432 &
python  ${path}/main.py 6433 &
python  ${path}/main.py 6434 &
python  ${path}/main.py 6435 &
python  ${path}/main.py 6436 &
python  ${path}/main.py 6437 &
python  ${path}/main.py 6438 &
python  ${path}/main.py 6439 &
python  ${path}/main.py 6440 &
python  ${path}/main.py 6441 &
python  ${path}/main.py 6442 &
python  ${path}/main.py 6443 &
wait
