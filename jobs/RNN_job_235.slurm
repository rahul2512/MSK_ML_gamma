#!/bin/bash -l
#SBATCH -J RNN_235
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8424 &
python  ${path}/main.py 8425 &
python  ${path}/main.py 8426 &
python  ${path}/main.py 8427 &
python  ${path}/main.py 8428 &
python  ${path}/main.py 8429 &
python  ${path}/main.py 8430 &
python  ${path}/main.py 8431 &
python  ${path}/main.py 8432 &
python  ${path}/main.py 8433 &
python  ${path}/main.py 8434 &
python  ${path}/main.py 8435 &
python  ${path}/main.py 8436 &
python  ${path}/main.py 8437 &
python  ${path}/main.py 8438 &
python  ${path}/main.py 8439 &
python  ${path}/main.py 8440 &
python  ${path}/main.py 8441 &
python  ${path}/main.py 8442 &
python  ${path}/main.py 8443 &
python  ${path}/main.py 8444 &
python  ${path}/main.py 8445 &
python  ${path}/main.py 8446 &
python  ${path}/main.py 8447 &
python  ${path}/main.py 8448 &
python  ${path}/main.py 8449 &
python  ${path}/main.py 8450 &
python  ${path}/main.py 8451 &
python  ${path}/main.py 8452 &
python  ${path}/main.py 8453 &
python  ${path}/main.py 8454 &
python  ${path}/main.py 8455 &
python  ${path}/main.py 8456 &
python  ${path}/main.py 8457 &
python  ${path}/main.py 8458 &
python  ${path}/main.py 8459 &
wait
