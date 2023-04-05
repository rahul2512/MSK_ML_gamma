#!/bin/bash -l
#SBATCH -J NN_92
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6552 &
python  ${path}/main.py 6553 &
python  ${path}/main.py 6554 &
python  ${path}/main.py 6555 &
python  ${path}/main.py 6556 &
python  ${path}/main.py 6557 &
python  ${path}/main.py 6558 &
python  ${path}/main.py 6559 &
python  ${path}/main.py 6560 &
python  ${path}/main.py 6561 &
python  ${path}/main.py 6562 &
python  ${path}/main.py 6563 &
python  ${path}/main.py 6564 &
python  ${path}/main.py 6565 &
python  ${path}/main.py 6566 &
python  ${path}/main.py 6567 &
python  ${path}/main.py 6568 &
python  ${path}/main.py 6569 &
python  ${path}/main.py 6570 &
python  ${path}/main.py 6571 &
python  ${path}/main.py 6572 &
python  ${path}/main.py 6573 &
python  ${path}/main.py 6574 &
python  ${path}/main.py 6575 &
python  ${path}/main.py 6576 &
python  ${path}/main.py 6577 &
python  ${path}/main.py 6578 &
python  ${path}/main.py 6579 &
python  ${path}/main.py 6580 &
python  ${path}/main.py 6581 &
python  ${path}/main.py 6582 &
python  ${path}/main.py 6583 &
python  ${path}/main.py 6584 &
python  ${path}/main.py 6585 &
python  ${path}/main.py 6586 &
python  ${path}/main.py 6587 &
python  ${path}/main.py 6588 &
python  ${path}/main.py 6589 &
python  ${path}/main.py 6590 &
python  ${path}/main.py 6591 &
python  ${path}/main.py 6592 &
python  ${path}/main.py 6593 &
python  ${path}/main.py 6594 &
python  ${path}/main.py 6595 &
python  ${path}/main.py 6596 &
python  ${path}/main.py 6597 &
python  ${path}/main.py 6598 &
python  ${path}/main.py 6599 &
python  ${path}/main.py 6600 &
python  ${path}/main.py 6601 &
python  ${path}/main.py 6602 &
python  ${path}/main.py 6603 &
python  ${path}/main.py 6604 &
python  ${path}/main.py 6605 &
python  ${path}/main.py 6606 &
python  ${path}/main.py 6607 &
python  ${path}/main.py 6608 &
python  ${path}/main.py 6609 &
python  ${path}/main.py 6610 &
python  ${path}/main.py 6611 &
python  ${path}/main.py 6612 &
python  ${path}/main.py 6613 &
python  ${path}/main.py 6614 &
python  ${path}/main.py 6615 &
python  ${path}/main.py 6616 &
python  ${path}/main.py 6617 &
python  ${path}/main.py 6618 &
python  ${path}/main.py 6619 &
python  ${path}/main.py 6620 &
python  ${path}/main.py 6621 &
python  ${path}/main.py 6622 &
python  ${path}/main.py 6623 &
wait
