#!/bin/bash -l
#SBATCH -J RNN_194
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6948 &
python  ${path}/main.py 6949 &
python  ${path}/main.py 6950 &
python  ${path}/main.py 6951 &
python  ${path}/main.py 6952 &
python  ${path}/main.py 6953 &
python  ${path}/main.py 6954 &
python  ${path}/main.py 6955 &
python  ${path}/main.py 6956 &
python  ${path}/main.py 6957 &
python  ${path}/main.py 6958 &
python  ${path}/main.py 6959 &
python  ${path}/main.py 6960 &
python  ${path}/main.py 6961 &
python  ${path}/main.py 6962 &
python  ${path}/main.py 6963 &
python  ${path}/main.py 6964 &
python  ${path}/main.py 6965 &
python  ${path}/main.py 6966 &
python  ${path}/main.py 6967 &
python  ${path}/main.py 6968 &
python  ${path}/main.py 6969 &
python  ${path}/main.py 6970 &
python  ${path}/main.py 6971 &
python  ${path}/main.py 6972 &
python  ${path}/main.py 6973 &
python  ${path}/main.py 6974 &
python  ${path}/main.py 6975 &
python  ${path}/main.py 6976 &
python  ${path}/main.py 6977 &
python  ${path}/main.py 6978 &
python  ${path}/main.py 6979 &
python  ${path}/main.py 6980 &
python  ${path}/main.py 6981 &
python  ${path}/main.py 6982 &
python  ${path}/main.py 6983 &
wait
