#!/bin/bash -l
#SBATCH -J RNN_77
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2736 &
python  ${path}/main.py 2737 &
python  ${path}/main.py 2738 &
python  ${path}/main.py 2739 &
python  ${path}/main.py 2740 &
python  ${path}/main.py 2741 &
python  ${path}/main.py 2742 &
python  ${path}/main.py 2743 &
python  ${path}/main.py 2744 &
python  ${path}/main.py 2745 &
python  ${path}/main.py 2746 &
python  ${path}/main.py 2747 &
python  ${path}/main.py 2748 &
python  ${path}/main.py 2749 &
python  ${path}/main.py 2750 &
python  ${path}/main.py 2751 &
python  ${path}/main.py 2752 &
python  ${path}/main.py 2753 &
python  ${path}/main.py 2754 &
python  ${path}/main.py 2755 &
python  ${path}/main.py 2756 &
python  ${path}/main.py 2757 &
python  ${path}/main.py 2758 &
python  ${path}/main.py 2759 &
python  ${path}/main.py 2760 &
python  ${path}/main.py 2761 &
python  ${path}/main.py 2762 &
python  ${path}/main.py 2763 &
python  ${path}/main.py 2764 &
python  ${path}/main.py 2765 &
python  ${path}/main.py 2766 &
python  ${path}/main.py 2767 &
python  ${path}/main.py 2768 &
python  ${path}/main.py 2769 &
python  ${path}/main.py 2770 &
python  ${path}/main.py 2771 &
wait
