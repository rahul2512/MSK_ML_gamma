#!/bin/bash -l
#SBATCH -J RNN_76
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2700 &
python  ${path}/main.py 2701 &
python  ${path}/main.py 2702 &
python  ${path}/main.py 2703 &
python  ${path}/main.py 2704 &
python  ${path}/main.py 2705 &
python  ${path}/main.py 2706 &
python  ${path}/main.py 2707 &
python  ${path}/main.py 2708 &
python  ${path}/main.py 2709 &
python  ${path}/main.py 2710 &
python  ${path}/main.py 2711 &
python  ${path}/main.py 2712 &
python  ${path}/main.py 2713 &
python  ${path}/main.py 2714 &
python  ${path}/main.py 2715 &
python  ${path}/main.py 2716 &
python  ${path}/main.py 2717 &
python  ${path}/main.py 2718 &
python  ${path}/main.py 2719 &
python  ${path}/main.py 2720 &
python  ${path}/main.py 2721 &
python  ${path}/main.py 2722 &
python  ${path}/main.py 2723 &
python  ${path}/main.py 2724 &
python  ${path}/main.py 2725 &
python  ${path}/main.py 2726 &
python  ${path}/main.py 2727 &
python  ${path}/main.py 2728 &
python  ${path}/main.py 2729 &
python  ${path}/main.py 2730 &
python  ${path}/main.py 2731 &
python  ${path}/main.py 2732 &
python  ${path}/main.py 2733 &
python  ${path}/main.py 2734 &
python  ${path}/main.py 2735 &
wait
