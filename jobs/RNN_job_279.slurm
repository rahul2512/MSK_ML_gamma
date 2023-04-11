#!/bin/bash -l
#SBATCH -J RNN_279
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10008 &
python  ${path}/main.py 10009 &
python  ${path}/main.py 10010 &
python  ${path}/main.py 10011 &
python  ${path}/main.py 10012 &
python  ${path}/main.py 10013 &
python  ${path}/main.py 10014 &
python  ${path}/main.py 10015 &
python  ${path}/main.py 10016 &
python  ${path}/main.py 10017 &
python  ${path}/main.py 10018 &
python  ${path}/main.py 10019 &
python  ${path}/main.py 10020 &
python  ${path}/main.py 10021 &
python  ${path}/main.py 10022 &
python  ${path}/main.py 10023 &
python  ${path}/main.py 10024 &
python  ${path}/main.py 10025 &
python  ${path}/main.py 10026 &
python  ${path}/main.py 10027 &
python  ${path}/main.py 10028 &
python  ${path}/main.py 10029 &
python  ${path}/main.py 10030 &
python  ${path}/main.py 10031 &
python  ${path}/main.py 10032 &
python  ${path}/main.py 10033 &
python  ${path}/main.py 10034 &
python  ${path}/main.py 10035 &
python  ${path}/main.py 10036 &
python  ${path}/main.py 10037 &
python  ${path}/main.py 10038 &
python  ${path}/main.py 10039 &
python  ${path}/main.py 10040 &
python  ${path}/main.py 10041 &
python  ${path}/main.py 10042 &
python  ${path}/main.py 10043 &
wait
