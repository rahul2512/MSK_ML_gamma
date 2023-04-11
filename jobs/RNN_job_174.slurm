#!/bin/bash -l
#SBATCH -J RNN_174
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6228 &
python  ${path}/main.py 6229 &
python  ${path}/main.py 6230 &
python  ${path}/main.py 6231 &
python  ${path}/main.py 6232 &
python  ${path}/main.py 6233 &
python  ${path}/main.py 6234 &
python  ${path}/main.py 6235 &
python  ${path}/main.py 6236 &
python  ${path}/main.py 6237 &
python  ${path}/main.py 6238 &
python  ${path}/main.py 6239 &
python  ${path}/main.py 6240 &
python  ${path}/main.py 6241 &
python  ${path}/main.py 6242 &
python  ${path}/main.py 6243 &
python  ${path}/main.py 6244 &
python  ${path}/main.py 6245 &
python  ${path}/main.py 6246 &
python  ${path}/main.py 6247 &
python  ${path}/main.py 6248 &
python  ${path}/main.py 6249 &
python  ${path}/main.py 6250 &
python  ${path}/main.py 6251 &
python  ${path}/main.py 6252 &
python  ${path}/main.py 6253 &
python  ${path}/main.py 6254 &
python  ${path}/main.py 6255 &
python  ${path}/main.py 6256 &
python  ${path}/main.py 6257 &
python  ${path}/main.py 6258 &
python  ${path}/main.py 6259 &
python  ${path}/main.py 6260 &
python  ${path}/main.py 6261 &
python  ${path}/main.py 6262 &
python  ${path}/main.py 6263 &
wait
