#!/bin/bash -l
#SBATCH -J RNN_287
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10296 &
python  ${path}/main.py 10297 &
python  ${path}/main.py 10298 &
python  ${path}/main.py 10299 &
python  ${path}/main.py 10300 &
python  ${path}/main.py 10301 &
python  ${path}/main.py 10302 &
python  ${path}/main.py 10303 &
python  ${path}/main.py 10304 &
python  ${path}/main.py 10305 &
python  ${path}/main.py 10306 &
python  ${path}/main.py 10307 &
python  ${path}/main.py 10308 &
python  ${path}/main.py 10309 &
python  ${path}/main.py 10310 &
python  ${path}/main.py 10311 &
python  ${path}/main.py 10312 &
python  ${path}/main.py 10313 &
python  ${path}/main.py 10314 &
python  ${path}/main.py 10315 &
python  ${path}/main.py 10316 &
python  ${path}/main.py 10317 &
python  ${path}/main.py 10318 &
python  ${path}/main.py 10319 &
python  ${path}/main.py 10320 &
python  ${path}/main.py 10321 &
python  ${path}/main.py 10322 &
python  ${path}/main.py 10323 &
python  ${path}/main.py 10324 &
python  ${path}/main.py 10325 &
python  ${path}/main.py 10326 &
python  ${path}/main.py 10327 &
python  ${path}/main.py 10328 &
python  ${path}/main.py 10329 &
python  ${path}/main.py 10330 &
python  ${path}/main.py 10331 &
wait
