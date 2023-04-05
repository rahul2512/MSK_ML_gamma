#!/bin/bash -l
#SBATCH -J NN_152
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10872 &
python  ${path}/main.py 10873 &
python  ${path}/main.py 10874 &
python  ${path}/main.py 10875 &
python  ${path}/main.py 10876 &
python  ${path}/main.py 10877 &
python  ${path}/main.py 10878 &
python  ${path}/main.py 10879 &
python  ${path}/main.py 10880 &
python  ${path}/main.py 10881 &
python  ${path}/main.py 10882 &
python  ${path}/main.py 10883 &
python  ${path}/main.py 10884 &
python  ${path}/main.py 10885 &
python  ${path}/main.py 10886 &
python  ${path}/main.py 10887 &
python  ${path}/main.py 10888 &
python  ${path}/main.py 10889 &
python  ${path}/main.py 10890 &
python  ${path}/main.py 10891 &
python  ${path}/main.py 10892 &
python  ${path}/main.py 10893 &
python  ${path}/main.py 10894 &
python  ${path}/main.py 10895 &
python  ${path}/main.py 10896 &
python  ${path}/main.py 10897 &
python  ${path}/main.py 10898 &
python  ${path}/main.py 10899 &
python  ${path}/main.py 10900 &
python  ${path}/main.py 10901 &
python  ${path}/main.py 10902 &
python  ${path}/main.py 10903 &
python  ${path}/main.py 10904 &
python  ${path}/main.py 10905 &
python  ${path}/main.py 10906 &
python  ${path}/main.py 10907 &
python  ${path}/main.py 10908 &
python  ${path}/main.py 10909 &
python  ${path}/main.py 10910 &
python  ${path}/main.py 10911 &
python  ${path}/main.py 10912 &
python  ${path}/main.py 10913 &
python  ${path}/main.py 10914 &
python  ${path}/main.py 10915 &
python  ${path}/main.py 10916 &
python  ${path}/main.py 10917 &
python  ${path}/main.py 10918 &
python  ${path}/main.py 10919 &
python  ${path}/main.py 10920 &
python  ${path}/main.py 10921 &
python  ${path}/main.py 10922 &
python  ${path}/main.py 10923 &
python  ${path}/main.py 10924 &
python  ${path}/main.py 10925 &
python  ${path}/main.py 10926 &
python  ${path}/main.py 10927 &
python  ${path}/main.py 10928 &
python  ${path}/main.py 10929 &
python  ${path}/main.py 10930 &
python  ${path}/main.py 10931 &
python  ${path}/main.py 10932 &
python  ${path}/main.py 10933 &
python  ${path}/main.py 10934 &
python  ${path}/main.py 10935 &
python  ${path}/main.py 10936 &
python  ${path}/main.py 10937 &
python  ${path}/main.py 10938 &
python  ${path}/main.py 10939 &
python  ${path}/main.py 10940 &
python  ${path}/main.py 10941 &
python  ${path}/main.py 10942 &
python  ${path}/main.py 10943 &
wait
