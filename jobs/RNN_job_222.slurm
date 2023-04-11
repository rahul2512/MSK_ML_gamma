#!/bin/bash -l
#SBATCH -J RNN_222
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7956 &
python  ${path}/main.py 7957 &
python  ${path}/main.py 7958 &
python  ${path}/main.py 7959 &
python  ${path}/main.py 7960 &
python  ${path}/main.py 7961 &
python  ${path}/main.py 7962 &
python  ${path}/main.py 7963 &
python  ${path}/main.py 7964 &
python  ${path}/main.py 7965 &
python  ${path}/main.py 7966 &
python  ${path}/main.py 7967 &
python  ${path}/main.py 7968 &
python  ${path}/main.py 7969 &
python  ${path}/main.py 7970 &
python  ${path}/main.py 7971 &
python  ${path}/main.py 7972 &
python  ${path}/main.py 7973 &
python  ${path}/main.py 7974 &
python  ${path}/main.py 7975 &
python  ${path}/main.py 7976 &
python  ${path}/main.py 7977 &
python  ${path}/main.py 7978 &
python  ${path}/main.py 7979 &
python  ${path}/main.py 7980 &
python  ${path}/main.py 7981 &
python  ${path}/main.py 7982 &
python  ${path}/main.py 7983 &
python  ${path}/main.py 7984 &
python  ${path}/main.py 7985 &
python  ${path}/main.py 7986 &
python  ${path}/main.py 7987 &
python  ${path}/main.py 7988 &
python  ${path}/main.py 7989 &
python  ${path}/main.py 7990 &
python  ${path}/main.py 7991 &
wait
