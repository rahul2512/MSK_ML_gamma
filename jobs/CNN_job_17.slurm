#!/bin/bash -l
#SBATCH -J CNN_17
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 384 &
python  ${path}/main.py 385 &
python  ${path}/main.py 386 &
python  ${path}/main.py 387 &
python  ${path}/main.py 388 &
python  ${path}/main.py 389 &
python  ${path}/main.py 390 &
python  ${path}/main.py 391 &
python  ${path}/main.py 392 &
python  ${path}/main.py 393 &
python  ${path}/main.py 394 &
python  ${path}/main.py 395 &
python  ${path}/main.py 396 &
python  ${path}/main.py 397 &
python  ${path}/main.py 398 &
python  ${path}/main.py 399 &
python  ${path}/main.py 400 &
python  ${path}/main.py 401 &
python  ${path}/main.py 402 &
python  ${path}/main.py 403 &
python  ${path}/main.py 404 &
python  ${path}/main.py 405 &
python  ${path}/main.py 406 &
python  ${path}/main.py 407 &
wait
