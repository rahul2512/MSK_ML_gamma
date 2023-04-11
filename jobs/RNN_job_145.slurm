#!/bin/bash -l
#SBATCH -J RNN_145
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5184 &
python  ${path}/main.py 5185 &
python  ${path}/main.py 5186 &
python  ${path}/main.py 5187 &
python  ${path}/main.py 5188 &
python  ${path}/main.py 5189 &
python  ${path}/main.py 5190 &
python  ${path}/main.py 5191 &
python  ${path}/main.py 5192 &
python  ${path}/main.py 5193 &
python  ${path}/main.py 5194 &
python  ${path}/main.py 5195 &
python  ${path}/main.py 5196 &
python  ${path}/main.py 5197 &
python  ${path}/main.py 5198 &
python  ${path}/main.py 5199 &
python  ${path}/main.py 5200 &
python  ${path}/main.py 5201 &
python  ${path}/main.py 5202 &
python  ${path}/main.py 5203 &
python  ${path}/main.py 5204 &
python  ${path}/main.py 5205 &
python  ${path}/main.py 5206 &
python  ${path}/main.py 5207 &
python  ${path}/main.py 5208 &
python  ${path}/main.py 5209 &
python  ${path}/main.py 5210 &
python  ${path}/main.py 5211 &
python  ${path}/main.py 5212 &
python  ${path}/main.py 5213 &
python  ${path}/main.py 5214 &
python  ${path}/main.py 5215 &
python  ${path}/main.py 5216 &
python  ${path}/main.py 5217 &
python  ${path}/main.py 5218 &
python  ${path}/main.py 5219 &
wait
