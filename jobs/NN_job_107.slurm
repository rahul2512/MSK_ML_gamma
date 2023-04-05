#!/bin/bash -l
#SBATCH -J NN_107
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7632 &
python  ${path}/main.py 7633 &
python  ${path}/main.py 7634 &
python  ${path}/main.py 7635 &
python  ${path}/main.py 7636 &
python  ${path}/main.py 7637 &
python  ${path}/main.py 7638 &
python  ${path}/main.py 7639 &
python  ${path}/main.py 7640 &
python  ${path}/main.py 7641 &
python  ${path}/main.py 7642 &
python  ${path}/main.py 7643 &
python  ${path}/main.py 7644 &
python  ${path}/main.py 7645 &
python  ${path}/main.py 7646 &
python  ${path}/main.py 7647 &
python  ${path}/main.py 7648 &
python  ${path}/main.py 7649 &
python  ${path}/main.py 7650 &
python  ${path}/main.py 7651 &
python  ${path}/main.py 7652 &
python  ${path}/main.py 7653 &
python  ${path}/main.py 7654 &
python  ${path}/main.py 7655 &
python  ${path}/main.py 7656 &
python  ${path}/main.py 7657 &
python  ${path}/main.py 7658 &
python  ${path}/main.py 7659 &
python  ${path}/main.py 7660 &
python  ${path}/main.py 7661 &
python  ${path}/main.py 7662 &
python  ${path}/main.py 7663 &
python  ${path}/main.py 7664 &
python  ${path}/main.py 7665 &
python  ${path}/main.py 7666 &
python  ${path}/main.py 7667 &
python  ${path}/main.py 7668 &
python  ${path}/main.py 7669 &
python  ${path}/main.py 7670 &
python  ${path}/main.py 7671 &
python  ${path}/main.py 7672 &
python  ${path}/main.py 7673 &
python  ${path}/main.py 7674 &
python  ${path}/main.py 7675 &
python  ${path}/main.py 7676 &
python  ${path}/main.py 7677 &
python  ${path}/main.py 7678 &
python  ${path}/main.py 7679 &
python  ${path}/main.py 7680 &
python  ${path}/main.py 7681 &
python  ${path}/main.py 7682 &
python  ${path}/main.py 7683 &
python  ${path}/main.py 7684 &
python  ${path}/main.py 7685 &
python  ${path}/main.py 7686 &
python  ${path}/main.py 7687 &
python  ${path}/main.py 7688 &
python  ${path}/main.py 7689 &
python  ${path}/main.py 7690 &
python  ${path}/main.py 7691 &
python  ${path}/main.py 7692 &
python  ${path}/main.py 7693 &
python  ${path}/main.py 7694 &
python  ${path}/main.py 7695 &
python  ${path}/main.py 7696 &
python  ${path}/main.py 7697 &
python  ${path}/main.py 7698 &
python  ${path}/main.py 7699 &
python  ${path}/main.py 7700 &
python  ${path}/main.py 7701 &
python  ${path}/main.py 7702 &
python  ${path}/main.py 7703 &
wait
