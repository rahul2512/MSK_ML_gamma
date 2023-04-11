#!/bin/bash -l
#SBATCH -J RNN_180
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6444 &
python  ${path}/main.py 6445 &
python  ${path}/main.py 6446 &
python  ${path}/main.py 6447 &
python  ${path}/main.py 6448 &
python  ${path}/main.py 6449 &
python  ${path}/main.py 6450 &
python  ${path}/main.py 6451 &
python  ${path}/main.py 6452 &
python  ${path}/main.py 6453 &
python  ${path}/main.py 6454 &
python  ${path}/main.py 6455 &
python  ${path}/main.py 6456 &
python  ${path}/main.py 6457 &
python  ${path}/main.py 6458 &
python  ${path}/main.py 6459 &
python  ${path}/main.py 6460 &
python  ${path}/main.py 6461 &
python  ${path}/main.py 6462 &
python  ${path}/main.py 6463 &
python  ${path}/main.py 6464 &
python  ${path}/main.py 6465 &
python  ${path}/main.py 6466 &
python  ${path}/main.py 6467 &
python  ${path}/main.py 6468 &
python  ${path}/main.py 6469 &
python  ${path}/main.py 6470 &
python  ${path}/main.py 6471 &
python  ${path}/main.py 6472 &
python  ${path}/main.py 6473 &
python  ${path}/main.py 6474 &
python  ${path}/main.py 6475 &
python  ${path}/main.py 6476 &
python  ${path}/main.py 6477 &
python  ${path}/main.py 6478 &
python  ${path}/main.py 6479 &
wait
