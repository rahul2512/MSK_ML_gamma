#!/bin/bash -l
#SBATCH -J RNN_137
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4896 &
python  ${path}/main.py 4897 &
python  ${path}/main.py 4898 &
python  ${path}/main.py 4899 &
python  ${path}/main.py 4900 &
python  ${path}/main.py 4901 &
python  ${path}/main.py 4902 &
python  ${path}/main.py 4903 &
python  ${path}/main.py 4904 &
python  ${path}/main.py 4905 &
python  ${path}/main.py 4906 &
python  ${path}/main.py 4907 &
python  ${path}/main.py 4908 &
python  ${path}/main.py 4909 &
python  ${path}/main.py 4910 &
python  ${path}/main.py 4911 &
python  ${path}/main.py 4912 &
python  ${path}/main.py 4913 &
python  ${path}/main.py 4914 &
python  ${path}/main.py 4915 &
python  ${path}/main.py 4916 &
python  ${path}/main.py 4917 &
python  ${path}/main.py 4918 &
python  ${path}/main.py 4919 &
python  ${path}/main.py 4920 &
python  ${path}/main.py 4921 &
python  ${path}/main.py 4922 &
python  ${path}/main.py 4923 &
python  ${path}/main.py 4924 &
python  ${path}/main.py 4925 &
python  ${path}/main.py 4926 &
python  ${path}/main.py 4927 &
python  ${path}/main.py 4928 &
python  ${path}/main.py 4929 &
python  ${path}/main.py 4930 &
python  ${path}/main.py 4931 &
wait
