#!/bin/bash -l
#SBATCH -J RNN_141
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5040 &
python  ${path}/main.py 5041 &
python  ${path}/main.py 5042 &
python  ${path}/main.py 5043 &
python  ${path}/main.py 5044 &
python  ${path}/main.py 5045 &
python  ${path}/main.py 5046 &
python  ${path}/main.py 5047 &
python  ${path}/main.py 5048 &
python  ${path}/main.py 5049 &
python  ${path}/main.py 5050 &
python  ${path}/main.py 5051 &
python  ${path}/main.py 5052 &
python  ${path}/main.py 5053 &
python  ${path}/main.py 5054 &
python  ${path}/main.py 5055 &
python  ${path}/main.py 5056 &
python  ${path}/main.py 5057 &
python  ${path}/main.py 5058 &
python  ${path}/main.py 5059 &
python  ${path}/main.py 5060 &
python  ${path}/main.py 5061 &
python  ${path}/main.py 5062 &
python  ${path}/main.py 5063 &
python  ${path}/main.py 5064 &
python  ${path}/main.py 5065 &
python  ${path}/main.py 5066 &
python  ${path}/main.py 5067 &
python  ${path}/main.py 5068 &
python  ${path}/main.py 5069 &
python  ${path}/main.py 5070 &
python  ${path}/main.py 5071 &
python  ${path}/main.py 5072 &
python  ${path}/main.py 5073 &
python  ${path}/main.py 5074 &
python  ${path}/main.py 5075 &
wait
