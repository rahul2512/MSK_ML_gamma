#!/bin/bash -l
#SBATCH -J CNN_20
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 456 &
python  ${path}/main.py 457 &
python  ${path}/main.py 458 &
python  ${path}/main.py 459 &
python  ${path}/main.py 460 &
python  ${path}/main.py 461 &
python  ${path}/main.py 462 &
python  ${path}/main.py 463 &
python  ${path}/main.py 464 &
python  ${path}/main.py 465 &
python  ${path}/main.py 466 &
python  ${path}/main.py 467 &
python  ${path}/main.py 468 &
python  ${path}/main.py 469 &
python  ${path}/main.py 470 &
python  ${path}/main.py 471 &
python  ${path}/main.py 472 &
python  ${path}/main.py 473 &
python  ${path}/main.py 474 &
python  ${path}/main.py 475 &
python  ${path}/main.py 476 &
python  ${path}/main.py 477 &
python  ${path}/main.py 478 &
python  ${path}/main.py 479 &
wait
