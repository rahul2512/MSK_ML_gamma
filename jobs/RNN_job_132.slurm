#!/bin/bash -l
#SBATCH -J RNN_132
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4716 &
python  ${path}/main.py 4717 &
python  ${path}/main.py 4718 &
python  ${path}/main.py 4719 &
python  ${path}/main.py 4720 &
python  ${path}/main.py 4721 &
python  ${path}/main.py 4722 &
python  ${path}/main.py 4723 &
python  ${path}/main.py 4724 &
python  ${path}/main.py 4725 &
python  ${path}/main.py 4726 &
python  ${path}/main.py 4727 &
python  ${path}/main.py 4728 &
python  ${path}/main.py 4729 &
python  ${path}/main.py 4730 &
python  ${path}/main.py 4731 &
python  ${path}/main.py 4732 &
python  ${path}/main.py 4733 &
python  ${path}/main.py 4734 &
python  ${path}/main.py 4735 &
python  ${path}/main.py 4736 &
python  ${path}/main.py 4737 &
python  ${path}/main.py 4738 &
python  ${path}/main.py 4739 &
python  ${path}/main.py 4740 &
python  ${path}/main.py 4741 &
python  ${path}/main.py 4742 &
python  ${path}/main.py 4743 &
python  ${path}/main.py 4744 &
python  ${path}/main.py 4745 &
python  ${path}/main.py 4746 &
python  ${path}/main.py 4747 &
python  ${path}/main.py 4748 &
python  ${path}/main.py 4749 &
python  ${path}/main.py 4750 &
python  ${path}/main.py 4751 &
wait
