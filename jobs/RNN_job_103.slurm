#!/bin/bash -l
#SBATCH -J RNN_103
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3672 &
python  ${path}/main.py 3673 &
python  ${path}/main.py 3674 &
python  ${path}/main.py 3675 &
python  ${path}/main.py 3676 &
python  ${path}/main.py 3677 &
python  ${path}/main.py 3678 &
python  ${path}/main.py 3679 &
python  ${path}/main.py 3680 &
python  ${path}/main.py 3681 &
python  ${path}/main.py 3682 &
python  ${path}/main.py 3683 &
python  ${path}/main.py 3684 &
python  ${path}/main.py 3685 &
python  ${path}/main.py 3686 &
python  ${path}/main.py 3687 &
python  ${path}/main.py 3688 &
python  ${path}/main.py 3689 &
python  ${path}/main.py 3690 &
python  ${path}/main.py 3691 &
python  ${path}/main.py 3692 &
python  ${path}/main.py 3693 &
python  ${path}/main.py 3694 &
python  ${path}/main.py 3695 &
python  ${path}/main.py 3696 &
python  ${path}/main.py 3697 &
python  ${path}/main.py 3698 &
python  ${path}/main.py 3699 &
python  ${path}/main.py 3700 &
python  ${path}/main.py 3701 &
python  ${path}/main.py 3702 &
python  ${path}/main.py 3703 &
python  ${path}/main.py 3704 &
python  ${path}/main.py 3705 &
python  ${path}/main.py 3706 &
python  ${path}/main.py 3707 &
wait
