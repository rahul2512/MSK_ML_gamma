#!/bin/bash -l
#SBATCH -J NN_78
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
python  ${path}/main.py 5580 &
python  ${path}/main.py 5581 &
python  ${path}/main.py 5582 &
python  ${path}/main.py 5583 &
python  ${path}/main.py 5584 &
python  ${path}/main.py 5585 &
python  ${path}/main.py 5586 &
python  ${path}/main.py 5587 &
python  ${path}/main.py 5588 &
python  ${path}/main.py 5589 &
python  ${path}/main.py 5590 &
python  ${path}/main.py 5591 &
python  ${path}/main.py 5592 &
python  ${path}/main.py 5593 &
python  ${path}/main.py 5594 &
python  ${path}/main.py 5595 &
python  ${path}/main.py 5596 &
python  ${path}/main.py 5597 &
python  ${path}/main.py 5598 &
python  ${path}/main.py 5599 &
python  ${path}/main.py 5600 &
python  ${path}/main.py 5601 &
python  ${path}/main.py 5602 &
python  ${path}/main.py 5603 &
python  ${path}/main.py 5604 &
python  ${path}/main.py 5605 &
python  ${path}/main.py 5606 &
python  ${path}/main.py 5607 &
python  ${path}/main.py 5608 &
python  ${path}/main.py 5609 &
python  ${path}/main.py 5610 &
python  ${path}/main.py 5611 &
python  ${path}/main.py 5612 &
python  ${path}/main.py 5613 &
python  ${path}/main.py 5614 &
python  ${path}/main.py 5615 &
wait
