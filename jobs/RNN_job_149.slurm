#!/bin/bash -l
#SBATCH -J RNN_149
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5328 &
python  ${path}/main.py 5329 &
python  ${path}/main.py 5330 &
python  ${path}/main.py 5331 &
python  ${path}/main.py 5332 &
python  ${path}/main.py 5333 &
python  ${path}/main.py 5334 &
python  ${path}/main.py 5335 &
python  ${path}/main.py 5336 &
python  ${path}/main.py 5337 &
python  ${path}/main.py 5338 &
python  ${path}/main.py 5339 &
python  ${path}/main.py 5340 &
python  ${path}/main.py 5341 &
python  ${path}/main.py 5342 &
python  ${path}/main.py 5343 &
python  ${path}/main.py 5344 &
python  ${path}/main.py 5345 &
python  ${path}/main.py 5346 &
python  ${path}/main.py 5347 &
python  ${path}/main.py 5348 &
python  ${path}/main.py 5349 &
python  ${path}/main.py 5350 &
python  ${path}/main.py 5351 &
python  ${path}/main.py 5352 &
python  ${path}/main.py 5353 &
python  ${path}/main.py 5354 &
python  ${path}/main.py 5355 &
python  ${path}/main.py 5356 &
python  ${path}/main.py 5357 &
python  ${path}/main.py 5358 &
python  ${path}/main.py 5359 &
python  ${path}/main.py 5360 &
python  ${path}/main.py 5361 &
python  ${path}/main.py 5362 &
python  ${path}/main.py 5363 &
wait
