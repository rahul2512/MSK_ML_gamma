#!/bin/bash -l
#SBATCH -J RNN_270
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9684 &
python  ${path}/main.py 9685 &
python  ${path}/main.py 9686 &
python  ${path}/main.py 9687 &
python  ${path}/main.py 9688 &
python  ${path}/main.py 9689 &
python  ${path}/main.py 9690 &
python  ${path}/main.py 9691 &
python  ${path}/main.py 9692 &
python  ${path}/main.py 9693 &
python  ${path}/main.py 9694 &
python  ${path}/main.py 9695 &
python  ${path}/main.py 9696 &
python  ${path}/main.py 9697 &
python  ${path}/main.py 9698 &
python  ${path}/main.py 9699 &
python  ${path}/main.py 9700 &
python  ${path}/main.py 9701 &
python  ${path}/main.py 9702 &
python  ${path}/main.py 9703 &
python  ${path}/main.py 9704 &
python  ${path}/main.py 9705 &
python  ${path}/main.py 9706 &
python  ${path}/main.py 9707 &
python  ${path}/main.py 9708 &
python  ${path}/main.py 9709 &
python  ${path}/main.py 9710 &
python  ${path}/main.py 9711 &
python  ${path}/main.py 9712 &
python  ${path}/main.py 9713 &
python  ${path}/main.py 9714 &
python  ${path}/main.py 9715 &
python  ${path}/main.py 9716 &
python  ${path}/main.py 9717 &
python  ${path}/main.py 9718 &
python  ${path}/main.py 9719 &
wait
