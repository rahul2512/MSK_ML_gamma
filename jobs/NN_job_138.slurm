#!/bin/bash -l
#SBATCH -J NN_138
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9864 &
python  ${path}/main.py 9865 &
python  ${path}/main.py 9866 &
python  ${path}/main.py 9867 &
python  ${path}/main.py 9868 &
python  ${path}/main.py 9869 &
python  ${path}/main.py 9870 &
python  ${path}/main.py 9871 &
python  ${path}/main.py 9872 &
python  ${path}/main.py 9873 &
python  ${path}/main.py 9874 &
python  ${path}/main.py 9875 &
python  ${path}/main.py 9876 &
python  ${path}/main.py 9877 &
python  ${path}/main.py 9878 &
python  ${path}/main.py 9879 &
python  ${path}/main.py 9880 &
python  ${path}/main.py 9881 &
python  ${path}/main.py 9882 &
python  ${path}/main.py 9883 &
python  ${path}/main.py 9884 &
python  ${path}/main.py 9885 &
python  ${path}/main.py 9886 &
python  ${path}/main.py 9887 &
python  ${path}/main.py 9888 &
python  ${path}/main.py 9889 &
python  ${path}/main.py 9890 &
python  ${path}/main.py 9891 &
python  ${path}/main.py 9892 &
python  ${path}/main.py 9893 &
python  ${path}/main.py 9894 &
python  ${path}/main.py 9895 &
python  ${path}/main.py 9896 &
python  ${path}/main.py 9897 &
python  ${path}/main.py 9898 &
python  ${path}/main.py 9899 &
python  ${path}/main.py 9900 &
python  ${path}/main.py 9901 &
python  ${path}/main.py 9902 &
python  ${path}/main.py 9903 &
python  ${path}/main.py 9904 &
python  ${path}/main.py 9905 &
python  ${path}/main.py 9906 &
python  ${path}/main.py 9907 &
python  ${path}/main.py 9908 &
python  ${path}/main.py 9909 &
python  ${path}/main.py 9910 &
python  ${path}/main.py 9911 &
python  ${path}/main.py 9912 &
python  ${path}/main.py 9913 &
python  ${path}/main.py 9914 &
python  ${path}/main.py 9915 &
python  ${path}/main.py 9916 &
python  ${path}/main.py 9917 &
python  ${path}/main.py 9918 &
python  ${path}/main.py 9919 &
python  ${path}/main.py 9920 &
python  ${path}/main.py 9921 &
python  ${path}/main.py 9922 &
python  ${path}/main.py 9923 &
python  ${path}/main.py 9924 &
python  ${path}/main.py 9925 &
python  ${path}/main.py 9926 &
python  ${path}/main.py 9927 &
python  ${path}/main.py 9928 &
python  ${path}/main.py 9929 &
python  ${path}/main.py 9930 &
python  ${path}/main.py 9931 &
python  ${path}/main.py 9932 &
python  ${path}/main.py 9933 &
python  ${path}/main.py 9934 &
python  ${path}/main.py 9935 &
wait
