#!/bin/bash -l
#SBATCH -J RNN_219
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7848 &
python  ${path}/main.py 7849 &
python  ${path}/main.py 7850 &
python  ${path}/main.py 7851 &
python  ${path}/main.py 7852 &
python  ${path}/main.py 7853 &
python  ${path}/main.py 7854 &
python  ${path}/main.py 7855 &
python  ${path}/main.py 7856 &
python  ${path}/main.py 7857 &
python  ${path}/main.py 7858 &
python  ${path}/main.py 7859 &
python  ${path}/main.py 7860 &
python  ${path}/main.py 7861 &
python  ${path}/main.py 7862 &
python  ${path}/main.py 7863 &
python  ${path}/main.py 7864 &
python  ${path}/main.py 7865 &
python  ${path}/main.py 7866 &
python  ${path}/main.py 7867 &
python  ${path}/main.py 7868 &
python  ${path}/main.py 7869 &
python  ${path}/main.py 7870 &
python  ${path}/main.py 7871 &
python  ${path}/main.py 7872 &
python  ${path}/main.py 7873 &
python  ${path}/main.py 7874 &
python  ${path}/main.py 7875 &
python  ${path}/main.py 7876 &
python  ${path}/main.py 7877 &
python  ${path}/main.py 7878 &
python  ${path}/main.py 7879 &
python  ${path}/main.py 7880 &
python  ${path}/main.py 7881 &
python  ${path}/main.py 7882 &
python  ${path}/main.py 7883 &
wait
