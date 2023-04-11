#!/bin/bash -l
#SBATCH -J RNN_18
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
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
python  ${path}/main.py 624 &
python  ${path}/main.py 625 &
python  ${path}/main.py 626 &
python  ${path}/main.py 627 &
python  ${path}/main.py 628 &
python  ${path}/main.py 629 &
python  ${path}/main.py 630 &
python  ${path}/main.py 631 &
python  ${path}/main.py 632 &
python  ${path}/main.py 633 &
python  ${path}/main.py 634 &
python  ${path}/main.py 635 &
python  ${path}/main.py 636 &
python  ${path}/main.py 637 &
python  ${path}/main.py 638 &
python  ${path}/main.py 639 &
python  ${path}/main.py 640 &
python  ${path}/main.py 641 &
python  ${path}/main.py 642 &
python  ${path}/main.py 643 &
python  ${path}/main.py 644 &
python  ${path}/main.py 645 &
python  ${path}/main.py 646 &
python  ${path}/main.py 647 &
wait
