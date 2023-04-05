#!/bin/bash -l
#SBATCH -J NN_98
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6984 &
python  ${path}/main.py 6985 &
python  ${path}/main.py 6986 &
python  ${path}/main.py 6987 &
python  ${path}/main.py 6988 &
python  ${path}/main.py 6989 &
python  ${path}/main.py 6990 &
python  ${path}/main.py 6991 &
python  ${path}/main.py 6992 &
python  ${path}/main.py 6993 &
python  ${path}/main.py 6994 &
python  ${path}/main.py 6995 &
python  ${path}/main.py 6996 &
python  ${path}/main.py 6997 &
python  ${path}/main.py 6998 &
python  ${path}/main.py 6999 &
python  ${path}/main.py 7000 &
python  ${path}/main.py 7001 &
python  ${path}/main.py 7002 &
python  ${path}/main.py 7003 &
python  ${path}/main.py 7004 &
python  ${path}/main.py 7005 &
python  ${path}/main.py 7006 &
python  ${path}/main.py 7007 &
python  ${path}/main.py 7008 &
python  ${path}/main.py 7009 &
python  ${path}/main.py 7010 &
python  ${path}/main.py 7011 &
python  ${path}/main.py 7012 &
python  ${path}/main.py 7013 &
python  ${path}/main.py 7014 &
python  ${path}/main.py 7015 &
python  ${path}/main.py 7016 &
python  ${path}/main.py 7017 &
python  ${path}/main.py 7018 &
python  ${path}/main.py 7019 &
python  ${path}/main.py 7020 &
python  ${path}/main.py 7021 &
python  ${path}/main.py 7022 &
python  ${path}/main.py 7023 &
python  ${path}/main.py 7024 &
python  ${path}/main.py 7025 &
python  ${path}/main.py 7026 &
python  ${path}/main.py 7027 &
python  ${path}/main.py 7028 &
python  ${path}/main.py 7029 &
python  ${path}/main.py 7030 &
python  ${path}/main.py 7031 &
python  ${path}/main.py 7032 &
python  ${path}/main.py 7033 &
python  ${path}/main.py 7034 &
python  ${path}/main.py 7035 &
python  ${path}/main.py 7036 &
python  ${path}/main.py 7037 &
python  ${path}/main.py 7038 &
python  ${path}/main.py 7039 &
python  ${path}/main.py 7040 &
python  ${path}/main.py 7041 &
python  ${path}/main.py 7042 &
python  ${path}/main.py 7043 &
python  ${path}/main.py 7044 &
python  ${path}/main.py 7045 &
python  ${path}/main.py 7046 &
python  ${path}/main.py 7047 &
python  ${path}/main.py 7048 &
python  ${path}/main.py 7049 &
python  ${path}/main.py 7050 &
python  ${path}/main.py 7051 &
python  ${path}/main.py 7052 &
python  ${path}/main.py 7053 &
python  ${path}/main.py 7054 &
python  ${path}/main.py 7055 &
wait
