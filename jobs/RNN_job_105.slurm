#!/bin/bash -l
#SBATCH -J RNN_105
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3744 &
python  ${path}/main.py 3745 &
python  ${path}/main.py 3746 &
python  ${path}/main.py 3747 &
python  ${path}/main.py 3748 &
python  ${path}/main.py 3749 &
python  ${path}/main.py 3750 &
python  ${path}/main.py 3751 &
python  ${path}/main.py 3752 &
python  ${path}/main.py 3753 &
python  ${path}/main.py 3754 &
python  ${path}/main.py 3755 &
python  ${path}/main.py 3756 &
python  ${path}/main.py 3757 &
python  ${path}/main.py 3758 &
python  ${path}/main.py 3759 &
python  ${path}/main.py 3760 &
python  ${path}/main.py 3761 &
python  ${path}/main.py 3762 &
python  ${path}/main.py 3763 &
python  ${path}/main.py 3764 &
python  ${path}/main.py 3765 &
python  ${path}/main.py 3766 &
python  ${path}/main.py 3767 &
python  ${path}/main.py 3768 &
python  ${path}/main.py 3769 &
python  ${path}/main.py 3770 &
python  ${path}/main.py 3771 &
python  ${path}/main.py 3772 &
python  ${path}/main.py 3773 &
python  ${path}/main.py 3774 &
python  ${path}/main.py 3775 &
python  ${path}/main.py 3776 &
python  ${path}/main.py 3777 &
python  ${path}/main.py 3778 &
python  ${path}/main.py 3779 &
wait
