#!/bin/bash -l
#SBATCH -J CNN_19
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 432 &
python  ${path}/main.py 433 &
python  ${path}/main.py 434 &
python  ${path}/main.py 435 &
python  ${path}/main.py 436 &
python  ${path}/main.py 437 &
python  ${path}/main.py 438 &
python  ${path}/main.py 439 &
python  ${path}/main.py 440 &
python  ${path}/main.py 441 &
python  ${path}/main.py 442 &
python  ${path}/main.py 443 &
python  ${path}/main.py 444 &
python  ${path}/main.py 445 &
python  ${path}/main.py 446 &
python  ${path}/main.py 447 &
python  ${path}/main.py 448 &
python  ${path}/main.py 449 &
python  ${path}/main.py 450 &
python  ${path}/main.py 451 &
python  ${path}/main.py 452 &
python  ${path}/main.py 453 &
python  ${path}/main.py 454 &
python  ${path}/main.py 455 &
wait
