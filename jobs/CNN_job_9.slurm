#!/bin/bash -l
#SBATCH -J CNN_9
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 192 &
python  ${path}/main.py 193 &
python  ${path}/main.py 194 &
python  ${path}/main.py 195 &
python  ${path}/main.py 196 &
python  ${path}/main.py 197 &
python  ${path}/main.py 198 &
python  ${path}/main.py 199 &
python  ${path}/main.py 200 &
python  ${path}/main.py 201 &
python  ${path}/main.py 202 &
python  ${path}/main.py 203 &
python  ${path}/main.py 204 &
python  ${path}/main.py 205 &
python  ${path}/main.py 206 &
python  ${path}/main.py 207 &
python  ${path}/main.py 208 &
python  ${path}/main.py 209 &
python  ${path}/main.py 210 &
python  ${path}/main.py 211 &
python  ${path}/main.py 212 &
python  ${path}/main.py 213 &
python  ${path}/main.py 214 &
python  ${path}/main.py 215 &
wait
