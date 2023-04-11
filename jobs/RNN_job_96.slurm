#!/bin/bash -l
#SBATCH -J RNN_96
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3420 &
python  ${path}/main.py 3421 &
python  ${path}/main.py 3422 &
python  ${path}/main.py 3423 &
python  ${path}/main.py 3424 &
python  ${path}/main.py 3425 &
python  ${path}/main.py 3426 &
python  ${path}/main.py 3427 &
python  ${path}/main.py 3428 &
python  ${path}/main.py 3429 &
python  ${path}/main.py 3430 &
python  ${path}/main.py 3431 &
python  ${path}/main.py 3432 &
python  ${path}/main.py 3433 &
python  ${path}/main.py 3434 &
python  ${path}/main.py 3435 &
python  ${path}/main.py 3436 &
python  ${path}/main.py 3437 &
python  ${path}/main.py 3438 &
python  ${path}/main.py 3439 &
python  ${path}/main.py 3440 &
python  ${path}/main.py 3441 &
python  ${path}/main.py 3442 &
python  ${path}/main.py 3443 &
python  ${path}/main.py 3444 &
python  ${path}/main.py 3445 &
python  ${path}/main.py 3446 &
python  ${path}/main.py 3447 &
python  ${path}/main.py 3448 &
python  ${path}/main.py 3449 &
python  ${path}/main.py 3450 &
python  ${path}/main.py 3451 &
python  ${path}/main.py 3452 &
python  ${path}/main.py 3453 &
python  ${path}/main.py 3454 &
python  ${path}/main.py 3455 &
wait
