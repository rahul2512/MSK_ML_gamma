#!/bin/bash -l
#SBATCH -J RNN_142
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5076 &
python  ${path}/main.py 5077 &
python  ${path}/main.py 5078 &
python  ${path}/main.py 5079 &
python  ${path}/main.py 5080 &
python  ${path}/main.py 5081 &
python  ${path}/main.py 5082 &
python  ${path}/main.py 5083 &
python  ${path}/main.py 5084 &
python  ${path}/main.py 5085 &
python  ${path}/main.py 5086 &
python  ${path}/main.py 5087 &
python  ${path}/main.py 5088 &
python  ${path}/main.py 5089 &
python  ${path}/main.py 5090 &
python  ${path}/main.py 5091 &
python  ${path}/main.py 5092 &
python  ${path}/main.py 5093 &
python  ${path}/main.py 5094 &
python  ${path}/main.py 5095 &
python  ${path}/main.py 5096 &
python  ${path}/main.py 5097 &
python  ${path}/main.py 5098 &
python  ${path}/main.py 5099 &
python  ${path}/main.py 5100 &
python  ${path}/main.py 5101 &
python  ${path}/main.py 5102 &
python  ${path}/main.py 5103 &
python  ${path}/main.py 5104 &
python  ${path}/main.py 5105 &
python  ${path}/main.py 5106 &
python  ${path}/main.py 5107 &
python  ${path}/main.py 5108 &
python  ${path}/main.py 5109 &
python  ${path}/main.py 5110 &
python  ${path}/main.py 5111 &
wait
