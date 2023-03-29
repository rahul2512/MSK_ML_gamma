#!/bin/bash -l
#SBATCH -J CNN_25
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 576 &
python  ${path}/main.py 577 &
python  ${path}/main.py 578 &
python  ${path}/main.py 579 &
python  ${path}/main.py 580 &
python  ${path}/main.py 581 &
python  ${path}/main.py 582 &
python  ${path}/main.py 583 &
python  ${path}/main.py 584 &
python  ${path}/main.py 585 &
python  ${path}/main.py 586 &
python  ${path}/main.py 587 &
python  ${path}/main.py 588 &
python  ${path}/main.py 589 &
python  ${path}/main.py 590 &
python  ${path}/main.py 591 &
python  ${path}/main.py 592 &
python  ${path}/main.py 593 &
python  ${path}/main.py 594 &
python  ${path}/main.py 595 &
python  ${path}/main.py 596 &
python  ${path}/main.py 597 &
python  ${path}/main.py 598 &
python  ${path}/main.py 599 &
wait
