#!/bin/bash -l
#SBATCH -J CNNLSTM_13
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 864 &
python  ${path}/main.py 865 &
python  ${path}/main.py 866 &
python  ${path}/main.py 867 &
python  ${path}/main.py 868 &
python  ${path}/main.py 869 &
python  ${path}/main.py 870 &
python  ${path}/main.py 871 &
python  ${path}/main.py 872 &
python  ${path}/main.py 873 &
python  ${path}/main.py 874 &
python  ${path}/main.py 875 &
python  ${path}/main.py 876 &
python  ${path}/main.py 877 &
python  ${path}/main.py 878 &
python  ${path}/main.py 879 &
python  ${path}/main.py 880 &
python  ${path}/main.py 881 &
python  ${path}/main.py 882 &
python  ${path}/main.py 883 &
python  ${path}/main.py 884 &
python  ${path}/main.py 885 &
python  ${path}/main.py 886 &
python  ${path}/main.py 887 &
python  ${path}/main.py 888 &
python  ${path}/main.py 889 &
python  ${path}/main.py 890 &
python  ${path}/main.py 891 &
python  ${path}/main.py 892 &
python  ${path}/main.py 893 &
python  ${path}/main.py 894 &
python  ${path}/main.py 895 &
python  ${path}/main.py 896 &
python  ${path}/main.py 897 &
python  ${path}/main.py 898 &
python  ${path}/main.py 899 &
python  ${path}/main.py 900 &
python  ${path}/main.py 901 &
python  ${path}/main.py 902 &
python  ${path}/main.py 903 &
python  ${path}/main.py 904 &
python  ${path}/main.py 905 &
python  ${path}/main.py 906 &
python  ${path}/main.py 907 &
python  ${path}/main.py 908 &
python  ${path}/main.py 909 &
python  ${path}/main.py 910 &
python  ${path}/main.py 911 &
python  ${path}/main.py 912 &
python  ${path}/main.py 913 &
python  ${path}/main.py 914 &
python  ${path}/main.py 915 &
python  ${path}/main.py 916 &
python  ${path}/main.py 917 &
python  ${path}/main.py 918 &
python  ${path}/main.py 919 &
python  ${path}/main.py 920 &
python  ${path}/main.py 921 &
python  ${path}/main.py 922 &
python  ${path}/main.py 923 &
python  ${path}/main.py 924 &
python  ${path}/main.py 925 &
python  ${path}/main.py 926 &
python  ${path}/main.py 927 &
python  ${path}/main.py 928 &
python  ${path}/main.py 929 &
python  ${path}/main.py 930 &
python  ${path}/main.py 931 &
python  ${path}/main.py 932 &
python  ${path}/main.py 933 &
python  ${path}/main.py 934 &
python  ${path}/main.py 935 &
wait
