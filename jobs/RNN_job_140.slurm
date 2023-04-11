#!/bin/bash -l
#SBATCH -J RNN_140
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5004 &
python  ${path}/main.py 5005 &
python  ${path}/main.py 5006 &
python  ${path}/main.py 5007 &
python  ${path}/main.py 5008 &
python  ${path}/main.py 5009 &
python  ${path}/main.py 5010 &
python  ${path}/main.py 5011 &
python  ${path}/main.py 5012 &
python  ${path}/main.py 5013 &
python  ${path}/main.py 5014 &
python  ${path}/main.py 5015 &
python  ${path}/main.py 5016 &
python  ${path}/main.py 5017 &
python  ${path}/main.py 5018 &
python  ${path}/main.py 5019 &
python  ${path}/main.py 5020 &
python  ${path}/main.py 5021 &
python  ${path}/main.py 5022 &
python  ${path}/main.py 5023 &
python  ${path}/main.py 5024 &
python  ${path}/main.py 5025 &
python  ${path}/main.py 5026 &
python  ${path}/main.py 5027 &
python  ${path}/main.py 5028 &
python  ${path}/main.py 5029 &
python  ${path}/main.py 5030 &
python  ${path}/main.py 5031 &
python  ${path}/main.py 5032 &
python  ${path}/main.py 5033 &
python  ${path}/main.py 5034 &
python  ${path}/main.py 5035 &
python  ${path}/main.py 5036 &
python  ${path}/main.py 5037 &
python  ${path}/main.py 5038 &
python  ${path}/main.py 5039 &
wait
