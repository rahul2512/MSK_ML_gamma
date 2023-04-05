#!/bin/bash -l
#SBATCH -J NN_111
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7920 &
python  ${path}/main.py 7921 &
python  ${path}/main.py 7922 &
python  ${path}/main.py 7923 &
python  ${path}/main.py 7924 &
python  ${path}/main.py 7925 &
python  ${path}/main.py 7926 &
python  ${path}/main.py 7927 &
python  ${path}/main.py 7928 &
python  ${path}/main.py 7929 &
python  ${path}/main.py 7930 &
python  ${path}/main.py 7931 &
python  ${path}/main.py 7932 &
python  ${path}/main.py 7933 &
python  ${path}/main.py 7934 &
python  ${path}/main.py 7935 &
python  ${path}/main.py 7936 &
python  ${path}/main.py 7937 &
python  ${path}/main.py 7938 &
python  ${path}/main.py 7939 &
python  ${path}/main.py 7940 &
python  ${path}/main.py 7941 &
python  ${path}/main.py 7942 &
python  ${path}/main.py 7943 &
python  ${path}/main.py 7944 &
python  ${path}/main.py 7945 &
python  ${path}/main.py 7946 &
python  ${path}/main.py 7947 &
python  ${path}/main.py 7948 &
python  ${path}/main.py 7949 &
python  ${path}/main.py 7950 &
python  ${path}/main.py 7951 &
python  ${path}/main.py 7952 &
python  ${path}/main.py 7953 &
python  ${path}/main.py 7954 &
python  ${path}/main.py 7955 &
python  ${path}/main.py 7956 &
python  ${path}/main.py 7957 &
python  ${path}/main.py 7958 &
python  ${path}/main.py 7959 &
python  ${path}/main.py 7960 &
python  ${path}/main.py 7961 &
python  ${path}/main.py 7962 &
python  ${path}/main.py 7963 &
python  ${path}/main.py 7964 &
python  ${path}/main.py 7965 &
python  ${path}/main.py 7966 &
python  ${path}/main.py 7967 &
python  ${path}/main.py 7968 &
python  ${path}/main.py 7969 &
python  ${path}/main.py 7970 &
python  ${path}/main.py 7971 &
python  ${path}/main.py 7972 &
python  ${path}/main.py 7973 &
python  ${path}/main.py 7974 &
python  ${path}/main.py 7975 &
python  ${path}/main.py 7976 &
python  ${path}/main.py 7977 &
python  ${path}/main.py 7978 &
python  ${path}/main.py 7979 &
python  ${path}/main.py 7980 &
python  ${path}/main.py 7981 &
python  ${path}/main.py 7982 &
python  ${path}/main.py 7983 &
python  ${path}/main.py 7984 &
python  ${path}/main.py 7985 &
python  ${path}/main.py 7986 &
python  ${path}/main.py 7987 &
python  ${path}/main.py 7988 &
python  ${path}/main.py 7989 &
python  ${path}/main.py 7990 &
python  ${path}/main.py 7991 &
wait
