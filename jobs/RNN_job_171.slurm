#!/bin/bash -l
#SBATCH -J RNN_171
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6120 &
python  ${path}/main.py 6121 &
python  ${path}/main.py 6122 &
python  ${path}/main.py 6123 &
python  ${path}/main.py 6124 &
python  ${path}/main.py 6125 &
python  ${path}/main.py 6126 &
python  ${path}/main.py 6127 &
python  ${path}/main.py 6128 &
python  ${path}/main.py 6129 &
python  ${path}/main.py 6130 &
python  ${path}/main.py 6131 &
python  ${path}/main.py 6132 &
python  ${path}/main.py 6133 &
python  ${path}/main.py 6134 &
python  ${path}/main.py 6135 &
python  ${path}/main.py 6136 &
python  ${path}/main.py 6137 &
python  ${path}/main.py 6138 &
python  ${path}/main.py 6139 &
python  ${path}/main.py 6140 &
python  ${path}/main.py 6141 &
python  ${path}/main.py 6142 &
python  ${path}/main.py 6143 &
python  ${path}/main.py 6144 &
python  ${path}/main.py 6145 &
python  ${path}/main.py 6146 &
python  ${path}/main.py 6147 &
python  ${path}/main.py 6148 &
python  ${path}/main.py 6149 &
python  ${path}/main.py 6150 &
python  ${path}/main.py 6151 &
python  ${path}/main.py 6152 &
python  ${path}/main.py 6153 &
python  ${path}/main.py 6154 &
python  ${path}/main.py 6155 &
wait
