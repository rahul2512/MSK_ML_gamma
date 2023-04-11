#!/bin/bash -l
#SBATCH -J RNN_154
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5508 &
python  ${path}/main.py 5509 &
python  ${path}/main.py 5510 &
python  ${path}/main.py 5511 &
python  ${path}/main.py 5512 &
python  ${path}/main.py 5513 &
python  ${path}/main.py 5514 &
python  ${path}/main.py 5515 &
python  ${path}/main.py 5516 &
python  ${path}/main.py 5517 &
python  ${path}/main.py 5518 &
python  ${path}/main.py 5519 &
python  ${path}/main.py 5520 &
python  ${path}/main.py 5521 &
python  ${path}/main.py 5522 &
python  ${path}/main.py 5523 &
python  ${path}/main.py 5524 &
python  ${path}/main.py 5525 &
python  ${path}/main.py 5526 &
python  ${path}/main.py 5527 &
python  ${path}/main.py 5528 &
python  ${path}/main.py 5529 &
python  ${path}/main.py 5530 &
python  ${path}/main.py 5531 &
python  ${path}/main.py 5532 &
python  ${path}/main.py 5533 &
python  ${path}/main.py 5534 &
python  ${path}/main.py 5535 &
python  ${path}/main.py 5536 &
python  ${path}/main.py 5537 &
python  ${path}/main.py 5538 &
python  ${path}/main.py 5539 &
python  ${path}/main.py 5540 &
python  ${path}/main.py 5541 &
python  ${path}/main.py 5542 &
python  ${path}/main.py 5543 &
wait
