#!/bin/bash -l
#SBATCH -J RNN_152
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5436 &
python  ${path}/main.py 5437 &
python  ${path}/main.py 5438 &
python  ${path}/main.py 5439 &
python  ${path}/main.py 5440 &
python  ${path}/main.py 5441 &
python  ${path}/main.py 5442 &
python  ${path}/main.py 5443 &
python  ${path}/main.py 5444 &
python  ${path}/main.py 5445 &
python  ${path}/main.py 5446 &
python  ${path}/main.py 5447 &
python  ${path}/main.py 5448 &
python  ${path}/main.py 5449 &
python  ${path}/main.py 5450 &
python  ${path}/main.py 5451 &
python  ${path}/main.py 5452 &
python  ${path}/main.py 5453 &
python  ${path}/main.py 5454 &
python  ${path}/main.py 5455 &
python  ${path}/main.py 5456 &
python  ${path}/main.py 5457 &
python  ${path}/main.py 5458 &
python  ${path}/main.py 5459 &
python  ${path}/main.py 5460 &
python  ${path}/main.py 5461 &
python  ${path}/main.py 5462 &
python  ${path}/main.py 5463 &
python  ${path}/main.py 5464 &
python  ${path}/main.py 5465 &
python  ${path}/main.py 5466 &
python  ${path}/main.py 5467 &
python  ${path}/main.py 5468 &
python  ${path}/main.py 5469 &
python  ${path}/main.py 5470 &
python  ${path}/main.py 5471 &
wait
