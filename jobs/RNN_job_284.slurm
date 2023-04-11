#!/bin/bash -l
#SBATCH -J RNN_284
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10188 &
python  ${path}/main.py 10189 &
python  ${path}/main.py 10190 &
python  ${path}/main.py 10191 &
python  ${path}/main.py 10192 &
python  ${path}/main.py 10193 &
python  ${path}/main.py 10194 &
python  ${path}/main.py 10195 &
python  ${path}/main.py 10196 &
python  ${path}/main.py 10197 &
python  ${path}/main.py 10198 &
python  ${path}/main.py 10199 &
python  ${path}/main.py 10200 &
python  ${path}/main.py 10201 &
python  ${path}/main.py 10202 &
python  ${path}/main.py 10203 &
python  ${path}/main.py 10204 &
python  ${path}/main.py 10205 &
python  ${path}/main.py 10206 &
python  ${path}/main.py 10207 &
python  ${path}/main.py 10208 &
python  ${path}/main.py 10209 &
python  ${path}/main.py 10210 &
python  ${path}/main.py 10211 &
python  ${path}/main.py 10212 &
python  ${path}/main.py 10213 &
python  ${path}/main.py 10214 &
python  ${path}/main.py 10215 &
python  ${path}/main.py 10216 &
python  ${path}/main.py 10217 &
python  ${path}/main.py 10218 &
python  ${path}/main.py 10219 &
python  ${path}/main.py 10220 &
python  ${path}/main.py 10221 &
python  ${path}/main.py 10222 &
python  ${path}/main.py 10223 &
wait
