#!/bin/bash -l
#SBATCH -J RNN_256
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9180 &
python  ${path}/main.py 9181 &
python  ${path}/main.py 9182 &
python  ${path}/main.py 9183 &
python  ${path}/main.py 9184 &
python  ${path}/main.py 9185 &
python  ${path}/main.py 9186 &
python  ${path}/main.py 9187 &
python  ${path}/main.py 9188 &
python  ${path}/main.py 9189 &
python  ${path}/main.py 9190 &
python  ${path}/main.py 9191 &
python  ${path}/main.py 9192 &
python  ${path}/main.py 9193 &
python  ${path}/main.py 9194 &
python  ${path}/main.py 9195 &
python  ${path}/main.py 9196 &
python  ${path}/main.py 9197 &
python  ${path}/main.py 9198 &
python  ${path}/main.py 9199 &
python  ${path}/main.py 9200 &
python  ${path}/main.py 9201 &
python  ${path}/main.py 9202 &
python  ${path}/main.py 9203 &
python  ${path}/main.py 9204 &
python  ${path}/main.py 9205 &
python  ${path}/main.py 9206 &
python  ${path}/main.py 9207 &
python  ${path}/main.py 9208 &
python  ${path}/main.py 9209 &
python  ${path}/main.py 9210 &
python  ${path}/main.py 9211 &
python  ${path}/main.py 9212 &
python  ${path}/main.py 9213 &
python  ${path}/main.py 9214 &
python  ${path}/main.py 9215 &
wait
