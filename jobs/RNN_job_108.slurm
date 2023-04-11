#!/bin/bash -l
#SBATCH -J RNN_108
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3852 &
python  ${path}/main.py 3853 &
python  ${path}/main.py 3854 &
python  ${path}/main.py 3855 &
python  ${path}/main.py 3856 &
python  ${path}/main.py 3857 &
python  ${path}/main.py 3858 &
python  ${path}/main.py 3859 &
python  ${path}/main.py 3860 &
python  ${path}/main.py 3861 &
python  ${path}/main.py 3862 &
python  ${path}/main.py 3863 &
python  ${path}/main.py 3864 &
python  ${path}/main.py 3865 &
python  ${path}/main.py 3866 &
python  ${path}/main.py 3867 &
python  ${path}/main.py 3868 &
python  ${path}/main.py 3869 &
python  ${path}/main.py 3870 &
python  ${path}/main.py 3871 &
python  ${path}/main.py 3872 &
python  ${path}/main.py 3873 &
python  ${path}/main.py 3874 &
python  ${path}/main.py 3875 &
python  ${path}/main.py 3876 &
python  ${path}/main.py 3877 &
python  ${path}/main.py 3878 &
python  ${path}/main.py 3879 &
python  ${path}/main.py 3880 &
python  ${path}/main.py 3881 &
python  ${path}/main.py 3882 &
python  ${path}/main.py 3883 &
python  ${path}/main.py 3884 &
python  ${path}/main.py 3885 &
python  ${path}/main.py 3886 &
python  ${path}/main.py 3887 &
wait
