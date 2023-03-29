#!/bin/bash -l
#SBATCH -J CNN_11
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 240 &
python  ${path}/main.py 241 &
python  ${path}/main.py 242 &
python  ${path}/main.py 243 &
python  ${path}/main.py 244 &
python  ${path}/main.py 245 &
python  ${path}/main.py 246 &
python  ${path}/main.py 247 &
python  ${path}/main.py 248 &
python  ${path}/main.py 249 &
python  ${path}/main.py 250 &
python  ${path}/main.py 251 &
python  ${path}/main.py 252 &
python  ${path}/main.py 253 &
python  ${path}/main.py 254 &
python  ${path}/main.py 255 &
python  ${path}/main.py 256 &
python  ${path}/main.py 257 &
python  ${path}/main.py 258 &
python  ${path}/main.py 259 &
python  ${path}/main.py 260 &
python  ${path}/main.py 261 &
python  ${path}/main.py 262 &
python  ${path}/main.py 263 &
wait
