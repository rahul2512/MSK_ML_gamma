#!/bin/bash -l
#SBATCH -J RNN_237
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8496 &
python  ${path}/main.py 8497 &
python  ${path}/main.py 8498 &
python  ${path}/main.py 8499 &
python  ${path}/main.py 8500 &
python  ${path}/main.py 8501 &
python  ${path}/main.py 8502 &
python  ${path}/main.py 8503 &
python  ${path}/main.py 8504 &
python  ${path}/main.py 8505 &
python  ${path}/main.py 8506 &
python  ${path}/main.py 8507 &
python  ${path}/main.py 8508 &
python  ${path}/main.py 8509 &
python  ${path}/main.py 8510 &
python  ${path}/main.py 8511 &
python  ${path}/main.py 8512 &
python  ${path}/main.py 8513 &
python  ${path}/main.py 8514 &
python  ${path}/main.py 8515 &
python  ${path}/main.py 8516 &
python  ${path}/main.py 8517 &
python  ${path}/main.py 8518 &
python  ${path}/main.py 8519 &
python  ${path}/main.py 8520 &
python  ${path}/main.py 8521 &
python  ${path}/main.py 8522 &
python  ${path}/main.py 8523 &
python  ${path}/main.py 8524 &
python  ${path}/main.py 8525 &
python  ${path}/main.py 8526 &
python  ${path}/main.py 8527 &
python  ${path}/main.py 8528 &
python  ${path}/main.py 8529 &
python  ${path}/main.py 8530 &
python  ${path}/main.py 8531 &
wait
