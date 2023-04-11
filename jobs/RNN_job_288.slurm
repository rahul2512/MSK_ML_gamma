#!/bin/bash -l
#SBATCH -J RNN_288
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10332 &
python  ${path}/main.py 10333 &
python  ${path}/main.py 10334 &
python  ${path}/main.py 10335 &
python  ${path}/main.py 10336 &
python  ${path}/main.py 10337 &
python  ${path}/main.py 10338 &
python  ${path}/main.py 10339 &
python  ${path}/main.py 10340 &
python  ${path}/main.py 10341 &
python  ${path}/main.py 10342 &
python  ${path}/main.py 10343 &
python  ${path}/main.py 10344 &
python  ${path}/main.py 10345 &
python  ${path}/main.py 10346 &
python  ${path}/main.py 10347 &
python  ${path}/main.py 10348 &
python  ${path}/main.py 10349 &
python  ${path}/main.py 10350 &
python  ${path}/main.py 10351 &
python  ${path}/main.py 10352 &
python  ${path}/main.py 10353 &
python  ${path}/main.py 10354 &
python  ${path}/main.py 10355 &
python  ${path}/main.py 10356 &
python  ${path}/main.py 10357 &
python  ${path}/main.py 10358 &
python  ${path}/main.py 10359 &
python  ${path}/main.py 10360 &
python  ${path}/main.py 10361 &
python  ${path}/main.py 10362 &
python  ${path}/main.py 10363 &
python  ${path}/main.py 10364 &
python  ${path}/main.py 10365 &
python  ${path}/main.py 10366 &
python  ${path}/main.py 10367 &
wait
