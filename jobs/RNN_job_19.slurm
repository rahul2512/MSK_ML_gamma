#!/bin/bash -l
#SBATCH -J RNN_19
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 648 &
python  ${path}/main.py 649 &
python  ${path}/main.py 650 &
python  ${path}/main.py 651 &
python  ${path}/main.py 652 &
python  ${path}/main.py 653 &
python  ${path}/main.py 654 &
python  ${path}/main.py 655 &
python  ${path}/main.py 656 &
python  ${path}/main.py 657 &
python  ${path}/main.py 658 &
python  ${path}/main.py 659 &
python  ${path}/main.py 660 &
python  ${path}/main.py 661 &
python  ${path}/main.py 662 &
python  ${path}/main.py 663 &
python  ${path}/main.py 664 &
python  ${path}/main.py 665 &
python  ${path}/main.py 666 &
python  ${path}/main.py 667 &
python  ${path}/main.py 668 &
python  ${path}/main.py 669 &
python  ${path}/main.py 670 &
python  ${path}/main.py 671 &
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
wait
