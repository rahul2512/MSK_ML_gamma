#!/bin/bash -l
#SBATCH -J NN_133
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9504 &
python  ${path}/main.py 9505 &
python  ${path}/main.py 9506 &
python  ${path}/main.py 9507 &
python  ${path}/main.py 9508 &
python  ${path}/main.py 9509 &
python  ${path}/main.py 9510 &
python  ${path}/main.py 9511 &
python  ${path}/main.py 9512 &
python  ${path}/main.py 9513 &
python  ${path}/main.py 9514 &
python  ${path}/main.py 9515 &
python  ${path}/main.py 9516 &
python  ${path}/main.py 9517 &
python  ${path}/main.py 9518 &
python  ${path}/main.py 9519 &
python  ${path}/main.py 9520 &
python  ${path}/main.py 9521 &
python  ${path}/main.py 9522 &
python  ${path}/main.py 9523 &
python  ${path}/main.py 9524 &
python  ${path}/main.py 9525 &
python  ${path}/main.py 9526 &
python  ${path}/main.py 9527 &
python  ${path}/main.py 9528 &
python  ${path}/main.py 9529 &
python  ${path}/main.py 9530 &
python  ${path}/main.py 9531 &
python  ${path}/main.py 9532 &
python  ${path}/main.py 9533 &
python  ${path}/main.py 9534 &
python  ${path}/main.py 9535 &
python  ${path}/main.py 9536 &
python  ${path}/main.py 9537 &
python  ${path}/main.py 9538 &
python  ${path}/main.py 9539 &
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
