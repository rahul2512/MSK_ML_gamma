#!/bin/bash -l
#SBATCH -J RNN_16
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 540 &
python  ${path}/main.py 541 &
python  ${path}/main.py 542 &
python  ${path}/main.py 543 &
python  ${path}/main.py 544 &
python  ${path}/main.py 545 &
python  ${path}/main.py 546 &
python  ${path}/main.py 547 &
python  ${path}/main.py 548 &
python  ${path}/main.py 549 &
python  ${path}/main.py 550 &
python  ${path}/main.py 551 &
python  ${path}/main.py 552 &
python  ${path}/main.py 553 &
python  ${path}/main.py 554 &
python  ${path}/main.py 555 &
python  ${path}/main.py 556 &
python  ${path}/main.py 557 &
python  ${path}/main.py 558 &
python  ${path}/main.py 559 &
python  ${path}/main.py 560 &
python  ${path}/main.py 561 &
python  ${path}/main.py 562 &
python  ${path}/main.py 563 &
python  ${path}/main.py 564 &
python  ${path}/main.py 565 &
python  ${path}/main.py 566 &
python  ${path}/main.py 567 &
python  ${path}/main.py 568 &
python  ${path}/main.py 569 &
python  ${path}/main.py 570 &
python  ${path}/main.py 571 &
python  ${path}/main.py 572 &
python  ${path}/main.py 573 &
python  ${path}/main.py 574 &
python  ${path}/main.py 575 &
wait
