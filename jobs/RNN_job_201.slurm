#!/bin/bash -l
#SBATCH -J RNN_201
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7200 &
python  ${path}/main.py 7201 &
python  ${path}/main.py 7202 &
python  ${path}/main.py 7203 &
python  ${path}/main.py 7204 &
python  ${path}/main.py 7205 &
python  ${path}/main.py 7206 &
python  ${path}/main.py 7207 &
python  ${path}/main.py 7208 &
python  ${path}/main.py 7209 &
python  ${path}/main.py 7210 &
python  ${path}/main.py 7211 &
python  ${path}/main.py 7212 &
python  ${path}/main.py 7213 &
python  ${path}/main.py 7214 &
python  ${path}/main.py 7215 &
python  ${path}/main.py 7216 &
python  ${path}/main.py 7217 &
python  ${path}/main.py 7218 &
python  ${path}/main.py 7219 &
python  ${path}/main.py 7220 &
python  ${path}/main.py 7221 &
python  ${path}/main.py 7222 &
python  ${path}/main.py 7223 &
python  ${path}/main.py 7224 &
python  ${path}/main.py 7225 &
python  ${path}/main.py 7226 &
python  ${path}/main.py 7227 &
python  ${path}/main.py 7228 &
python  ${path}/main.py 7229 &
python  ${path}/main.py 7230 &
python  ${path}/main.py 7231 &
python  ${path}/main.py 7232 &
python  ${path}/main.py 7233 &
python  ${path}/main.py 7234 &
python  ${path}/main.py 7235 &
wait
