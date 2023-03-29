#!/bin/bash -l
#SBATCH -J CNN_29
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 672 &
python  ${path}/main.py 673 &
python  ${path}/main.py 674 &
python  ${path}/main.py 675 &
python  ${path}/main.py 676 &
python  ${path}/main.py 677 &
python  ${path}/main.py 678 &
python  ${path}/main.py 679 &
python  ${path}/main.py 680 &
python  ${path}/main.py 681 &
python  ${path}/main.py 682 &
python  ${path}/main.py 683 &
python  ${path}/main.py 684 &
python  ${path}/main.py 685 &
python  ${path}/main.py 686 &
python  ${path}/main.py 687 &
python  ${path}/main.py 688 &
python  ${path}/main.py 689 &
python  ${path}/main.py 690 &
python  ${path}/main.py 691 &
python  ${path}/main.py 692 &
python  ${path}/main.py 693 &
python  ${path}/main.py 694 &
python  ${path}/main.py 695 &
wait
