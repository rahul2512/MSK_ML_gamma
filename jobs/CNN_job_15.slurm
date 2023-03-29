#!/bin/bash -l
#SBATCH -J CNN_15
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 336 &
python  ${path}/main.py 337 &
python  ${path}/main.py 338 &
python  ${path}/main.py 339 &
python  ${path}/main.py 340 &
python  ${path}/main.py 341 &
python  ${path}/main.py 342 &
python  ${path}/main.py 343 &
python  ${path}/main.py 344 &
python  ${path}/main.py 345 &
python  ${path}/main.py 346 &
python  ${path}/main.py 347 &
python  ${path}/main.py 348 &
python  ${path}/main.py 349 &
python  ${path}/main.py 350 &
python  ${path}/main.py 351 &
python  ${path}/main.py 352 &
python  ${path}/main.py 353 &
python  ${path}/main.py 354 &
python  ${path}/main.py 355 &
python  ${path}/main.py 356 &
python  ${path}/main.py 357 &
python  ${path}/main.py 358 &
python  ${path}/main.py 359 &
wait
