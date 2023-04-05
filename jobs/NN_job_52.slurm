#!/bin/bash -l
#SBATCH -J NN_52
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
python  ${path}/main.py 3708 &
python  ${path}/main.py 3709 &
python  ${path}/main.py 3710 &
python  ${path}/main.py 3711 &
python  ${path}/main.py 3712 &
python  ${path}/main.py 3713 &
python  ${path}/main.py 3714 &
python  ${path}/main.py 3715 &
python  ${path}/main.py 3716 &
python  ${path}/main.py 3717 &
python  ${path}/main.py 3718 &
python  ${path}/main.py 3719 &
python  ${path}/main.py 3720 &
python  ${path}/main.py 3721 &
python  ${path}/main.py 3722 &
python  ${path}/main.py 3723 &
python  ${path}/main.py 3724 &
python  ${path}/main.py 3725 &
python  ${path}/main.py 3726 &
python  ${path}/main.py 3727 &
python  ${path}/main.py 3728 &
python  ${path}/main.py 3729 &
python  ${path}/main.py 3730 &
python  ${path}/main.py 3731 &
python  ${path}/main.py 3732 &
python  ${path}/main.py 3733 &
python  ${path}/main.py 3734 &
python  ${path}/main.py 3735 &
python  ${path}/main.py 3736 &
python  ${path}/main.py 3737 &
python  ${path}/main.py 3738 &
python  ${path}/main.py 3739 &
python  ${path}/main.py 3740 &
python  ${path}/main.py 3741 &
python  ${path}/main.py 3742 &
python  ${path}/main.py 3743 &
wait
