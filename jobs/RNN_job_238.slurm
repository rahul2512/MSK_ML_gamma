#!/bin/bash -l
#SBATCH -J RNN_238
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8532 &
python  ${path}/main.py 8533 &
python  ${path}/main.py 8534 &
python  ${path}/main.py 8535 &
python  ${path}/main.py 8536 &
python  ${path}/main.py 8537 &
python  ${path}/main.py 8538 &
python  ${path}/main.py 8539 &
python  ${path}/main.py 8540 &
python  ${path}/main.py 8541 &
python  ${path}/main.py 8542 &
python  ${path}/main.py 8543 &
python  ${path}/main.py 8544 &
python  ${path}/main.py 8545 &
python  ${path}/main.py 8546 &
python  ${path}/main.py 8547 &
python  ${path}/main.py 8548 &
python  ${path}/main.py 8549 &
python  ${path}/main.py 8550 &
python  ${path}/main.py 8551 &
python  ${path}/main.py 8552 &
python  ${path}/main.py 8553 &
python  ${path}/main.py 8554 &
python  ${path}/main.py 8555 &
python  ${path}/main.py 8556 &
python  ${path}/main.py 8557 &
python  ${path}/main.py 8558 &
python  ${path}/main.py 8559 &
python  ${path}/main.py 8560 &
python  ${path}/main.py 8561 &
python  ${path}/main.py 8562 &
python  ${path}/main.py 8563 &
python  ${path}/main.py 8564 &
python  ${path}/main.py 8565 &
python  ${path}/main.py 8566 &
python  ${path}/main.py 8567 &
wait
