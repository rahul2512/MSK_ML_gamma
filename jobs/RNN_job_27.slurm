#!/bin/bash -l
#SBATCH -J RNN_27
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 936 &
python  ${path}/main.py 937 &
python  ${path}/main.py 938 &
python  ${path}/main.py 939 &
python  ${path}/main.py 940 &
python  ${path}/main.py 941 &
python  ${path}/main.py 942 &
python  ${path}/main.py 943 &
python  ${path}/main.py 944 &
python  ${path}/main.py 945 &
python  ${path}/main.py 946 &
python  ${path}/main.py 947 &
python  ${path}/main.py 948 &
python  ${path}/main.py 949 &
python  ${path}/main.py 950 &
python  ${path}/main.py 951 &
python  ${path}/main.py 952 &
python  ${path}/main.py 953 &
python  ${path}/main.py 954 &
python  ${path}/main.py 955 &
python  ${path}/main.py 956 &
python  ${path}/main.py 957 &
python  ${path}/main.py 958 &
python  ${path}/main.py 959 &
python  ${path}/main.py 960 &
python  ${path}/main.py 961 &
python  ${path}/main.py 962 &
python  ${path}/main.py 963 &
python  ${path}/main.py 964 &
python  ${path}/main.py 965 &
python  ${path}/main.py 966 &
python  ${path}/main.py 967 &
python  ${path}/main.py 968 &
python  ${path}/main.py 969 &
python  ${path}/main.py 970 &
python  ${path}/main.py 971 &
wait
