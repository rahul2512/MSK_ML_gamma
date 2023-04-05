#!/bin/bash -l
#SBATCH -J NN_93
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6624 &
python  ${path}/main.py 6625 &
python  ${path}/main.py 6626 &
python  ${path}/main.py 6627 &
python  ${path}/main.py 6628 &
python  ${path}/main.py 6629 &
python  ${path}/main.py 6630 &
python  ${path}/main.py 6631 &
python  ${path}/main.py 6632 &
python  ${path}/main.py 6633 &
python  ${path}/main.py 6634 &
python  ${path}/main.py 6635 &
python  ${path}/main.py 6636 &
python  ${path}/main.py 6637 &
python  ${path}/main.py 6638 &
python  ${path}/main.py 6639 &
python  ${path}/main.py 6640 &
python  ${path}/main.py 6641 &
python  ${path}/main.py 6642 &
python  ${path}/main.py 6643 &
python  ${path}/main.py 6644 &
python  ${path}/main.py 6645 &
python  ${path}/main.py 6646 &
python  ${path}/main.py 6647 &
python  ${path}/main.py 6648 &
python  ${path}/main.py 6649 &
python  ${path}/main.py 6650 &
python  ${path}/main.py 6651 &
python  ${path}/main.py 6652 &
python  ${path}/main.py 6653 &
python  ${path}/main.py 6654 &
python  ${path}/main.py 6655 &
python  ${path}/main.py 6656 &
python  ${path}/main.py 6657 &
python  ${path}/main.py 6658 &
python  ${path}/main.py 6659 &
python  ${path}/main.py 6660 &
python  ${path}/main.py 6661 &
python  ${path}/main.py 6662 &
python  ${path}/main.py 6663 &
python  ${path}/main.py 6664 &
python  ${path}/main.py 6665 &
python  ${path}/main.py 6666 &
python  ${path}/main.py 6667 &
python  ${path}/main.py 6668 &
python  ${path}/main.py 6669 &
python  ${path}/main.py 6670 &
python  ${path}/main.py 6671 &
python  ${path}/main.py 6672 &
python  ${path}/main.py 6673 &
python  ${path}/main.py 6674 &
python  ${path}/main.py 6675 &
python  ${path}/main.py 6676 &
python  ${path}/main.py 6677 &
python  ${path}/main.py 6678 &
python  ${path}/main.py 6679 &
python  ${path}/main.py 6680 &
python  ${path}/main.py 6681 &
python  ${path}/main.py 6682 &
python  ${path}/main.py 6683 &
python  ${path}/main.py 6684 &
python  ${path}/main.py 6685 &
python  ${path}/main.py 6686 &
python  ${path}/main.py 6687 &
python  ${path}/main.py 6688 &
python  ${path}/main.py 6689 &
python  ${path}/main.py 6690 &
python  ${path}/main.py 6691 &
python  ${path}/main.py 6692 &
python  ${path}/main.py 6693 &
python  ${path}/main.py 6694 &
python  ${path}/main.py 6695 &
wait
