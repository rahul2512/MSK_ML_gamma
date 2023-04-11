#!/bin/bash -l
#SBATCH -J RNN_186
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6660 &
python  ${path}/main.py 6661 &
python  ${path}/main.py 6662 &
python  ${path}/main.py 6663 &
python  ${path}/main.py 6664 &
python  ${path}/main.py 6665 &
python  ${path}/main.py 6666 &
python  ${path}/main.py 6667 &
python  ${path}/main.py 6668 &
python  ${path}/main.py 6669 &
python  ${path}/main.py 6670 &
python  ${path}/main.py 6671 &
python  ${path}/main.py 6672 &
python  ${path}/main.py 6673 &
python  ${path}/main.py 6674 &
python  ${path}/main.py 6675 &
python  ${path}/main.py 6676 &
python  ${path}/main.py 6677 &
python  ${path}/main.py 6678 &
python  ${path}/main.py 6679 &
python  ${path}/main.py 6680 &
python  ${path}/main.py 6681 &
python  ${path}/main.py 6682 &
python  ${path}/main.py 6683 &
python  ${path}/main.py 6684 &
python  ${path}/main.py 6685 &
python  ${path}/main.py 6686 &
python  ${path}/main.py 6687 &
python  ${path}/main.py 6688 &
python  ${path}/main.py 6689 &
python  ${path}/main.py 6690 &
python  ${path}/main.py 6691 &
python  ${path}/main.py 6692 &
python  ${path}/main.py 6693 &
python  ${path}/main.py 6694 &
python  ${path}/main.py 6695 &
wait
