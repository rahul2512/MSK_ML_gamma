#!/bin/bash -l
#SBATCH -J RNN_266
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9540 &
python  ${path}/main.py 9541 &
python  ${path}/main.py 9542 &
python  ${path}/main.py 9543 &
python  ${path}/main.py 9544 &
python  ${path}/main.py 9545 &
python  ${path}/main.py 9546 &
python  ${path}/main.py 9547 &
python  ${path}/main.py 9548 &
python  ${path}/main.py 9549 &
python  ${path}/main.py 9550 &
python  ${path}/main.py 9551 &
python  ${path}/main.py 9552 &
python  ${path}/main.py 9553 &
python  ${path}/main.py 9554 &
python  ${path}/main.py 9555 &
python  ${path}/main.py 9556 &
python  ${path}/main.py 9557 &
python  ${path}/main.py 9558 &
python  ${path}/main.py 9559 &
python  ${path}/main.py 9560 &
python  ${path}/main.py 9561 &
python  ${path}/main.py 9562 &
python  ${path}/main.py 9563 &
python  ${path}/main.py 9564 &
python  ${path}/main.py 9565 &
python  ${path}/main.py 9566 &
python  ${path}/main.py 9567 &
python  ${path}/main.py 9568 &
python  ${path}/main.py 9569 &
python  ${path}/main.py 9570 &
python  ${path}/main.py 9571 &
python  ${path}/main.py 9572 &
python  ${path}/main.py 9573 &
python  ${path}/main.py 9574 &
python  ${path}/main.py 9575 &
wait
