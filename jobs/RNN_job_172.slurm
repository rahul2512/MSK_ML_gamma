#!/bin/bash -l
#SBATCH -J RNN_172
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6156 &
python  ${path}/main.py 6157 &
python  ${path}/main.py 6158 &
python  ${path}/main.py 6159 &
python  ${path}/main.py 6160 &
python  ${path}/main.py 6161 &
python  ${path}/main.py 6162 &
python  ${path}/main.py 6163 &
python  ${path}/main.py 6164 &
python  ${path}/main.py 6165 &
python  ${path}/main.py 6166 &
python  ${path}/main.py 6167 &
python  ${path}/main.py 6168 &
python  ${path}/main.py 6169 &
python  ${path}/main.py 6170 &
python  ${path}/main.py 6171 &
python  ${path}/main.py 6172 &
python  ${path}/main.py 6173 &
python  ${path}/main.py 6174 &
python  ${path}/main.py 6175 &
python  ${path}/main.py 6176 &
python  ${path}/main.py 6177 &
python  ${path}/main.py 6178 &
python  ${path}/main.py 6179 &
python  ${path}/main.py 6180 &
python  ${path}/main.py 6181 &
python  ${path}/main.py 6182 &
python  ${path}/main.py 6183 &
python  ${path}/main.py 6184 &
python  ${path}/main.py 6185 &
python  ${path}/main.py 6186 &
python  ${path}/main.py 6187 &
python  ${path}/main.py 6188 &
python  ${path}/main.py 6189 &
python  ${path}/main.py 6190 &
python  ${path}/main.py 6191 &
wait
