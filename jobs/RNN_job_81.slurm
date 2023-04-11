#!/bin/bash -l
#SBATCH -J RNN_81
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2880 &
python  ${path}/main.py 2881 &
python  ${path}/main.py 2882 &
python  ${path}/main.py 2883 &
python  ${path}/main.py 2884 &
python  ${path}/main.py 2885 &
python  ${path}/main.py 2886 &
python  ${path}/main.py 2887 &
python  ${path}/main.py 2888 &
python  ${path}/main.py 2889 &
python  ${path}/main.py 2890 &
python  ${path}/main.py 2891 &
python  ${path}/main.py 2892 &
python  ${path}/main.py 2893 &
python  ${path}/main.py 2894 &
python  ${path}/main.py 2895 &
python  ${path}/main.py 2896 &
python  ${path}/main.py 2897 &
python  ${path}/main.py 2898 &
python  ${path}/main.py 2899 &
python  ${path}/main.py 2900 &
python  ${path}/main.py 2901 &
python  ${path}/main.py 2902 &
python  ${path}/main.py 2903 &
python  ${path}/main.py 2904 &
python  ${path}/main.py 2905 &
python  ${path}/main.py 2906 &
python  ${path}/main.py 2907 &
python  ${path}/main.py 2908 &
python  ${path}/main.py 2909 &
python  ${path}/main.py 2910 &
python  ${path}/main.py 2911 &
python  ${path}/main.py 2912 &
python  ${path}/main.py 2913 &
python  ${path}/main.py 2914 &
python  ${path}/main.py 2915 &
wait
