#!/bin/bash -l
#SBATCH -J NN_112
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7992 &
python  ${path}/main.py 7993 &
python  ${path}/main.py 7994 &
python  ${path}/main.py 7995 &
python  ${path}/main.py 7996 &
python  ${path}/main.py 7997 &
python  ${path}/main.py 7998 &
python  ${path}/main.py 7999 &
python  ${path}/main.py 8000 &
python  ${path}/main.py 8001 &
python  ${path}/main.py 8002 &
python  ${path}/main.py 8003 &
python  ${path}/main.py 8004 &
python  ${path}/main.py 8005 &
python  ${path}/main.py 8006 &
python  ${path}/main.py 8007 &
python  ${path}/main.py 8008 &
python  ${path}/main.py 8009 &
python  ${path}/main.py 8010 &
python  ${path}/main.py 8011 &
python  ${path}/main.py 8012 &
python  ${path}/main.py 8013 &
python  ${path}/main.py 8014 &
python  ${path}/main.py 8015 &
python  ${path}/main.py 8016 &
python  ${path}/main.py 8017 &
python  ${path}/main.py 8018 &
python  ${path}/main.py 8019 &
python  ${path}/main.py 8020 &
python  ${path}/main.py 8021 &
python  ${path}/main.py 8022 &
python  ${path}/main.py 8023 &
python  ${path}/main.py 8024 &
python  ${path}/main.py 8025 &
python  ${path}/main.py 8026 &
python  ${path}/main.py 8027 &
python  ${path}/main.py 8028 &
python  ${path}/main.py 8029 &
python  ${path}/main.py 8030 &
python  ${path}/main.py 8031 &
python  ${path}/main.py 8032 &
python  ${path}/main.py 8033 &
python  ${path}/main.py 8034 &
python  ${path}/main.py 8035 &
python  ${path}/main.py 8036 &
python  ${path}/main.py 8037 &
python  ${path}/main.py 8038 &
python  ${path}/main.py 8039 &
python  ${path}/main.py 8040 &
python  ${path}/main.py 8041 &
python  ${path}/main.py 8042 &
python  ${path}/main.py 8043 &
python  ${path}/main.py 8044 &
python  ${path}/main.py 8045 &
python  ${path}/main.py 8046 &
python  ${path}/main.py 8047 &
python  ${path}/main.py 8048 &
python  ${path}/main.py 8049 &
python  ${path}/main.py 8050 &
python  ${path}/main.py 8051 &
python  ${path}/main.py 8052 &
python  ${path}/main.py 8053 &
python  ${path}/main.py 8054 &
python  ${path}/main.py 8055 &
python  ${path}/main.py 8056 &
python  ${path}/main.py 8057 &
python  ${path}/main.py 8058 &
python  ${path}/main.py 8059 &
python  ${path}/main.py 8060 &
python  ${path}/main.py 8061 &
python  ${path}/main.py 8062 &
python  ${path}/main.py 8063 &
wait
