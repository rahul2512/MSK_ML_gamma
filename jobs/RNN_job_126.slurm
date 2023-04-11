#!/bin/bash -l
#SBATCH -J RNN_126
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4500 &
python  ${path}/main.py 4501 &
python  ${path}/main.py 4502 &
python  ${path}/main.py 4503 &
python  ${path}/main.py 4504 &
python  ${path}/main.py 4505 &
python  ${path}/main.py 4506 &
python  ${path}/main.py 4507 &
python  ${path}/main.py 4508 &
python  ${path}/main.py 4509 &
python  ${path}/main.py 4510 &
python  ${path}/main.py 4511 &
python  ${path}/main.py 4512 &
python  ${path}/main.py 4513 &
python  ${path}/main.py 4514 &
python  ${path}/main.py 4515 &
python  ${path}/main.py 4516 &
python  ${path}/main.py 4517 &
python  ${path}/main.py 4518 &
python  ${path}/main.py 4519 &
python  ${path}/main.py 4520 &
python  ${path}/main.py 4521 &
python  ${path}/main.py 4522 &
python  ${path}/main.py 4523 &
python  ${path}/main.py 4524 &
python  ${path}/main.py 4525 &
python  ${path}/main.py 4526 &
python  ${path}/main.py 4527 &
python  ${path}/main.py 4528 &
python  ${path}/main.py 4529 &
python  ${path}/main.py 4530 &
python  ${path}/main.py 4531 &
python  ${path}/main.py 4532 &
python  ${path}/main.py 4533 &
python  ${path}/main.py 4534 &
python  ${path}/main.py 4535 &
wait
