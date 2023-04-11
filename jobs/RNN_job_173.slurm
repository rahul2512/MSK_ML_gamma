#!/bin/bash -l
#SBATCH -J RNN_173
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6192 &
python  ${path}/main.py 6193 &
python  ${path}/main.py 6194 &
python  ${path}/main.py 6195 &
python  ${path}/main.py 6196 &
python  ${path}/main.py 6197 &
python  ${path}/main.py 6198 &
python  ${path}/main.py 6199 &
python  ${path}/main.py 6200 &
python  ${path}/main.py 6201 &
python  ${path}/main.py 6202 &
python  ${path}/main.py 6203 &
python  ${path}/main.py 6204 &
python  ${path}/main.py 6205 &
python  ${path}/main.py 6206 &
python  ${path}/main.py 6207 &
python  ${path}/main.py 6208 &
python  ${path}/main.py 6209 &
python  ${path}/main.py 6210 &
python  ${path}/main.py 6211 &
python  ${path}/main.py 6212 &
python  ${path}/main.py 6213 &
python  ${path}/main.py 6214 &
python  ${path}/main.py 6215 &
python  ${path}/main.py 6216 &
python  ${path}/main.py 6217 &
python  ${path}/main.py 6218 &
python  ${path}/main.py 6219 &
python  ${path}/main.py 6220 &
python  ${path}/main.py 6221 &
python  ${path}/main.py 6222 &
python  ${path}/main.py 6223 &
python  ${path}/main.py 6224 &
python  ${path}/main.py 6225 &
python  ${path}/main.py 6226 &
python  ${path}/main.py 6227 &
wait
