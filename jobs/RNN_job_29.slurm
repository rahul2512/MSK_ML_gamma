#!/bin/bash -l
#SBATCH -J RNN_29
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1008 &
python  ${path}/main.py 1009 &
python  ${path}/main.py 1010 &
python  ${path}/main.py 1011 &
python  ${path}/main.py 1012 &
python  ${path}/main.py 1013 &
python  ${path}/main.py 1014 &
python  ${path}/main.py 1015 &
python  ${path}/main.py 1016 &
python  ${path}/main.py 1017 &
python  ${path}/main.py 1018 &
python  ${path}/main.py 1019 &
python  ${path}/main.py 1020 &
python  ${path}/main.py 1021 &
python  ${path}/main.py 1022 &
python  ${path}/main.py 1023 &
python  ${path}/main.py 1024 &
python  ${path}/main.py 1025 &
python  ${path}/main.py 1026 &
python  ${path}/main.py 1027 &
python  ${path}/main.py 1028 &
python  ${path}/main.py 1029 &
python  ${path}/main.py 1030 &
python  ${path}/main.py 1031 &
python  ${path}/main.py 1032 &
python  ${path}/main.py 1033 &
python  ${path}/main.py 1034 &
python  ${path}/main.py 1035 &
python  ${path}/main.py 1036 &
python  ${path}/main.py 1037 &
python  ${path}/main.py 1038 &
python  ${path}/main.py 1039 &
python  ${path}/main.py 1040 &
python  ${path}/main.py 1041 &
python  ${path}/main.py 1042 &
python  ${path}/main.py 1043 &
wait
