#!/bin/bash -l
#SBATCH -J NN_153
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10944 &
python  ${path}/main.py 10945 &
python  ${path}/main.py 10946 &
python  ${path}/main.py 10947 &
python  ${path}/main.py 10948 &
python  ${path}/main.py 10949 &
python  ${path}/main.py 10950 &
python  ${path}/main.py 10951 &
python  ${path}/main.py 10952 &
python  ${path}/main.py 10953 &
python  ${path}/main.py 10954 &
python  ${path}/main.py 10955 &
python  ${path}/main.py 10956 &
python  ${path}/main.py 10957 &
python  ${path}/main.py 10958 &
python  ${path}/main.py 10959 &
python  ${path}/main.py 10960 &
python  ${path}/main.py 10961 &
python  ${path}/main.py 10962 &
python  ${path}/main.py 10963 &
python  ${path}/main.py 10964 &
python  ${path}/main.py 10965 &
python  ${path}/main.py 10966 &
python  ${path}/main.py 10967 &
python  ${path}/main.py 10968 &
python  ${path}/main.py 10969 &
python  ${path}/main.py 10970 &
python  ${path}/main.py 10971 &
python  ${path}/main.py 10972 &
python  ${path}/main.py 10973 &
python  ${path}/main.py 10974 &
python  ${path}/main.py 10975 &
python  ${path}/main.py 10976 &
python  ${path}/main.py 10977 &
python  ${path}/main.py 10978 &
python  ${path}/main.py 10979 &
python  ${path}/main.py 10980 &
python  ${path}/main.py 10981 &
python  ${path}/main.py 10982 &
python  ${path}/main.py 10983 &
python  ${path}/main.py 10984 &
python  ${path}/main.py 10985 &
python  ${path}/main.py 10986 &
python  ${path}/main.py 10987 &
python  ${path}/main.py 10988 &
python  ${path}/main.py 10989 &
python  ${path}/main.py 10990 &
python  ${path}/main.py 10991 &
python  ${path}/main.py 10992 &
python  ${path}/main.py 10993 &
python  ${path}/main.py 10994 &
python  ${path}/main.py 10995 &
python  ${path}/main.py 10996 &
python  ${path}/main.py 10997 &
python  ${path}/main.py 10998 &
python  ${path}/main.py 10999 &
python  ${path}/main.py 11000 &
python  ${path}/main.py 11001 &
python  ${path}/main.py 11002 &
python  ${path}/main.py 11003 &
python  ${path}/main.py 11004 &
python  ${path}/main.py 11005 &
python  ${path}/main.py 11006 &
python  ${path}/main.py 11007 &
python  ${path}/main.py 11008 &
python  ${path}/main.py 11009 &
python  ${path}/main.py 11010 &
python  ${path}/main.py 11011 &
python  ${path}/main.py 11012 &
python  ${path}/main.py 11013 &
python  ${path}/main.py 11014 &
python  ${path}/main.py 11015 &
wait
