#!/bin/bash -l
#SBATCH -J NN_140
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
python  ${path}/main.py 10044 &
python  ${path}/main.py 10045 &
python  ${path}/main.py 10046 &
python  ${path}/main.py 10047 &
python  ${path}/main.py 10048 &
python  ${path}/main.py 10049 &
python  ${path}/main.py 10050 &
python  ${path}/main.py 10051 &
python  ${path}/main.py 10052 &
python  ${path}/main.py 10053 &
python  ${path}/main.py 10054 &
python  ${path}/main.py 10055 &
python  ${path}/main.py 10056 &
python  ${path}/main.py 10057 &
python  ${path}/main.py 10058 &
python  ${path}/main.py 10059 &
python  ${path}/main.py 10060 &
python  ${path}/main.py 10061 &
python  ${path}/main.py 10062 &
python  ${path}/main.py 10063 &
python  ${path}/main.py 10064 &
python  ${path}/main.py 10065 &
python  ${path}/main.py 10066 &
python  ${path}/main.py 10067 &
python  ${path}/main.py 10068 &
python  ${path}/main.py 10069 &
python  ${path}/main.py 10070 &
python  ${path}/main.py 10071 &
python  ${path}/main.py 10072 &
python  ${path}/main.py 10073 &
python  ${path}/main.py 10074 &
python  ${path}/main.py 10075 &
python  ${path}/main.py 10076 &
python  ${path}/main.py 10077 &
python  ${path}/main.py 10078 &
python  ${path}/main.py 10079 &
wait
