#!/bin/bash -l
#SBATCH -J RNN_176
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6300 &
python  ${path}/main.py 6301 &
python  ${path}/main.py 6302 &
python  ${path}/main.py 6303 &
python  ${path}/main.py 6304 &
python  ${path}/main.py 6305 &
python  ${path}/main.py 6306 &
python  ${path}/main.py 6307 &
python  ${path}/main.py 6308 &
python  ${path}/main.py 6309 &
python  ${path}/main.py 6310 &
python  ${path}/main.py 6311 &
python  ${path}/main.py 6312 &
python  ${path}/main.py 6313 &
python  ${path}/main.py 6314 &
python  ${path}/main.py 6315 &
python  ${path}/main.py 6316 &
python  ${path}/main.py 6317 &
python  ${path}/main.py 6318 &
python  ${path}/main.py 6319 &
python  ${path}/main.py 6320 &
python  ${path}/main.py 6321 &
python  ${path}/main.py 6322 &
python  ${path}/main.py 6323 &
python  ${path}/main.py 6324 &
python  ${path}/main.py 6325 &
python  ${path}/main.py 6326 &
python  ${path}/main.py 6327 &
python  ${path}/main.py 6328 &
python  ${path}/main.py 6329 &
python  ${path}/main.py 6330 &
python  ${path}/main.py 6331 &
python  ${path}/main.py 6332 &
python  ${path}/main.py 6333 &
python  ${path}/main.py 6334 &
python  ${path}/main.py 6335 &
wait
