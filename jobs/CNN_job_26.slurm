#!/bin/bash -l
#SBATCH -J CNN_26
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 600 &
python  ${path}/main.py 601 &
python  ${path}/main.py 602 &
python  ${path}/main.py 603 &
python  ${path}/main.py 604 &
python  ${path}/main.py 605 &
python  ${path}/main.py 606 &
python  ${path}/main.py 607 &
python  ${path}/main.py 608 &
python  ${path}/main.py 609 &
python  ${path}/main.py 610 &
python  ${path}/main.py 611 &
python  ${path}/main.py 612 &
python  ${path}/main.py 613 &
python  ${path}/main.py 614 &
python  ${path}/main.py 615 &
python  ${path}/main.py 616 &
python  ${path}/main.py 617 &
python  ${path}/main.py 618 &
python  ${path}/main.py 619 &
python  ${path}/main.py 620 &
python  ${path}/main.py 621 &
python  ${path}/main.py 622 &
python  ${path}/main.py 623 &
wait
