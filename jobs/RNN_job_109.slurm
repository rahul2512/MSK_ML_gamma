#!/bin/bash -l
#SBATCH -J RNN_109
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3888 &
python  ${path}/main.py 3889 &
python  ${path}/main.py 3890 &
python  ${path}/main.py 3891 &
python  ${path}/main.py 3892 &
python  ${path}/main.py 3893 &
python  ${path}/main.py 3894 &
python  ${path}/main.py 3895 &
python  ${path}/main.py 3896 &
python  ${path}/main.py 3897 &
python  ${path}/main.py 3898 &
python  ${path}/main.py 3899 &
python  ${path}/main.py 3900 &
python  ${path}/main.py 3901 &
python  ${path}/main.py 3902 &
python  ${path}/main.py 3903 &
python  ${path}/main.py 3904 &
python  ${path}/main.py 3905 &
python  ${path}/main.py 3906 &
python  ${path}/main.py 3907 &
python  ${path}/main.py 3908 &
python  ${path}/main.py 3909 &
python  ${path}/main.py 3910 &
python  ${path}/main.py 3911 &
python  ${path}/main.py 3912 &
python  ${path}/main.py 3913 &
python  ${path}/main.py 3914 &
python  ${path}/main.py 3915 &
python  ${path}/main.py 3916 &
python  ${path}/main.py 3917 &
python  ${path}/main.py 3918 &
python  ${path}/main.py 3919 &
python  ${path}/main.py 3920 &
python  ${path}/main.py 3921 &
python  ${path}/main.py 3922 &
python  ${path}/main.py 3923 &
wait
