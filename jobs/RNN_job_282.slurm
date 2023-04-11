#!/bin/bash -l
#SBATCH -J RNN_282
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10116 &
python  ${path}/main.py 10117 &
python  ${path}/main.py 10118 &
python  ${path}/main.py 10119 &
python  ${path}/main.py 10120 &
python  ${path}/main.py 10121 &
python  ${path}/main.py 10122 &
python  ${path}/main.py 10123 &
python  ${path}/main.py 10124 &
python  ${path}/main.py 10125 &
python  ${path}/main.py 10126 &
python  ${path}/main.py 10127 &
python  ${path}/main.py 10128 &
python  ${path}/main.py 10129 &
python  ${path}/main.py 10130 &
python  ${path}/main.py 10131 &
python  ${path}/main.py 10132 &
python  ${path}/main.py 10133 &
python  ${path}/main.py 10134 &
python  ${path}/main.py 10135 &
python  ${path}/main.py 10136 &
python  ${path}/main.py 10137 &
python  ${path}/main.py 10138 &
python  ${path}/main.py 10139 &
python  ${path}/main.py 10140 &
python  ${path}/main.py 10141 &
python  ${path}/main.py 10142 &
python  ${path}/main.py 10143 &
python  ${path}/main.py 10144 &
python  ${path}/main.py 10145 &
python  ${path}/main.py 10146 &
python  ${path}/main.py 10147 &
python  ${path}/main.py 10148 &
python  ${path}/main.py 10149 &
python  ${path}/main.py 10150 &
python  ${path}/main.py 10151 &
wait
