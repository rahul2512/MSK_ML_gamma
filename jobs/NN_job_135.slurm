#!/bin/bash -l
#SBATCH -J NN_135
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9648 &
python  ${path}/main.py 9649 &
python  ${path}/main.py 9650 &
python  ${path}/main.py 9651 &
python  ${path}/main.py 9652 &
python  ${path}/main.py 9653 &
python  ${path}/main.py 9654 &
python  ${path}/main.py 9655 &
python  ${path}/main.py 9656 &
python  ${path}/main.py 9657 &
python  ${path}/main.py 9658 &
python  ${path}/main.py 9659 &
python  ${path}/main.py 9660 &
python  ${path}/main.py 9661 &
python  ${path}/main.py 9662 &
python  ${path}/main.py 9663 &
python  ${path}/main.py 9664 &
python  ${path}/main.py 9665 &
python  ${path}/main.py 9666 &
python  ${path}/main.py 9667 &
python  ${path}/main.py 9668 &
python  ${path}/main.py 9669 &
python  ${path}/main.py 9670 &
python  ${path}/main.py 9671 &
python  ${path}/main.py 9672 &
python  ${path}/main.py 9673 &
python  ${path}/main.py 9674 &
python  ${path}/main.py 9675 &
python  ${path}/main.py 9676 &
python  ${path}/main.py 9677 &
python  ${path}/main.py 9678 &
python  ${path}/main.py 9679 &
python  ${path}/main.py 9680 &
python  ${path}/main.py 9681 &
python  ${path}/main.py 9682 &
python  ${path}/main.py 9683 &
python  ${path}/main.py 9684 &
python  ${path}/main.py 9685 &
python  ${path}/main.py 9686 &
python  ${path}/main.py 9687 &
python  ${path}/main.py 9688 &
python  ${path}/main.py 9689 &
python  ${path}/main.py 9690 &
python  ${path}/main.py 9691 &
python  ${path}/main.py 9692 &
python  ${path}/main.py 9693 &
python  ${path}/main.py 9694 &
python  ${path}/main.py 9695 &
python  ${path}/main.py 9696 &
python  ${path}/main.py 9697 &
python  ${path}/main.py 9698 &
python  ${path}/main.py 9699 &
python  ${path}/main.py 9700 &
python  ${path}/main.py 9701 &
python  ${path}/main.py 9702 &
python  ${path}/main.py 9703 &
python  ${path}/main.py 9704 &
python  ${path}/main.py 9705 &
python  ${path}/main.py 9706 &
python  ${path}/main.py 9707 &
python  ${path}/main.py 9708 &
python  ${path}/main.py 9709 &
python  ${path}/main.py 9710 &
python  ${path}/main.py 9711 &
python  ${path}/main.py 9712 &
python  ${path}/main.py 9713 &
python  ${path}/main.py 9714 &
python  ${path}/main.py 9715 &
python  ${path}/main.py 9716 &
python  ${path}/main.py 9717 &
python  ${path}/main.py 9718 &
python  ${path}/main.py 9719 &
wait
