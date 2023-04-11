#!/bin/bash -l
#SBATCH -J RNN_181
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6480 &
python  ${path}/main.py 6481 &
python  ${path}/main.py 6482 &
python  ${path}/main.py 6483 &
python  ${path}/main.py 6484 &
python  ${path}/main.py 6485 &
python  ${path}/main.py 6486 &
python  ${path}/main.py 6487 &
python  ${path}/main.py 6488 &
python  ${path}/main.py 6489 &
python  ${path}/main.py 6490 &
python  ${path}/main.py 6491 &
python  ${path}/main.py 6492 &
python  ${path}/main.py 6493 &
python  ${path}/main.py 6494 &
python  ${path}/main.py 6495 &
python  ${path}/main.py 6496 &
python  ${path}/main.py 6497 &
python  ${path}/main.py 6498 &
python  ${path}/main.py 6499 &
python  ${path}/main.py 6500 &
python  ${path}/main.py 6501 &
python  ${path}/main.py 6502 &
python  ${path}/main.py 6503 &
python  ${path}/main.py 6504 &
python  ${path}/main.py 6505 &
python  ${path}/main.py 6506 &
python  ${path}/main.py 6507 &
python  ${path}/main.py 6508 &
python  ${path}/main.py 6509 &
python  ${path}/main.py 6510 &
python  ${path}/main.py 6511 &
python  ${path}/main.py 6512 &
python  ${path}/main.py 6513 &
python  ${path}/main.py 6514 &
python  ${path}/main.py 6515 &
wait
