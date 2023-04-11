#!/bin/bash -l
#SBATCH -J RNN_155
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5544 &
python  ${path}/main.py 5545 &
python  ${path}/main.py 5546 &
python  ${path}/main.py 5547 &
python  ${path}/main.py 5548 &
python  ${path}/main.py 5549 &
python  ${path}/main.py 5550 &
python  ${path}/main.py 5551 &
python  ${path}/main.py 5552 &
python  ${path}/main.py 5553 &
python  ${path}/main.py 5554 &
python  ${path}/main.py 5555 &
python  ${path}/main.py 5556 &
python  ${path}/main.py 5557 &
python  ${path}/main.py 5558 &
python  ${path}/main.py 5559 &
python  ${path}/main.py 5560 &
python  ${path}/main.py 5561 &
python  ${path}/main.py 5562 &
python  ${path}/main.py 5563 &
python  ${path}/main.py 5564 &
python  ${path}/main.py 5565 &
python  ${path}/main.py 5566 &
python  ${path}/main.py 5567 &
python  ${path}/main.py 5568 &
python  ${path}/main.py 5569 &
python  ${path}/main.py 5570 &
python  ${path}/main.py 5571 &
python  ${path}/main.py 5572 &
python  ${path}/main.py 5573 &
python  ${path}/main.py 5574 &
python  ${path}/main.py 5575 &
python  ${path}/main.py 5576 &
python  ${path}/main.py 5577 &
python  ${path}/main.py 5578 &
python  ${path}/main.py 5579 &
wait
