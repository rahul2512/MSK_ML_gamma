#!/bin/bash -l
#SBATCH -J RNN_239
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8568 &
python  ${path}/main.py 8569 &
python  ${path}/main.py 8570 &
python  ${path}/main.py 8571 &
python  ${path}/main.py 8572 &
python  ${path}/main.py 8573 &
python  ${path}/main.py 8574 &
python  ${path}/main.py 8575 &
python  ${path}/main.py 8576 &
python  ${path}/main.py 8577 &
python  ${path}/main.py 8578 &
python  ${path}/main.py 8579 &
python  ${path}/main.py 8580 &
python  ${path}/main.py 8581 &
python  ${path}/main.py 8582 &
python  ${path}/main.py 8583 &
python  ${path}/main.py 8584 &
python  ${path}/main.py 8585 &
python  ${path}/main.py 8586 &
python  ${path}/main.py 8587 &
python  ${path}/main.py 8588 &
python  ${path}/main.py 8589 &
python  ${path}/main.py 8590 &
python  ${path}/main.py 8591 &
python  ${path}/main.py 8592 &
python  ${path}/main.py 8593 &
python  ${path}/main.py 8594 &
python  ${path}/main.py 8595 &
python  ${path}/main.py 8596 &
python  ${path}/main.py 8597 &
python  ${path}/main.py 8598 &
python  ${path}/main.py 8599 &
python  ${path}/main.py 8600 &
python  ${path}/main.py 8601 &
python  ${path}/main.py 8602 &
python  ${path}/main.py 8603 &
wait
