#!/bin/bash -l
#SBATCH -J NN_56
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3960 &
python  ${path}/main.py 3961 &
python  ${path}/main.py 3962 &
python  ${path}/main.py 3963 &
python  ${path}/main.py 3964 &
python  ${path}/main.py 3965 &
python  ${path}/main.py 3966 &
python  ${path}/main.py 3967 &
python  ${path}/main.py 3968 &
python  ${path}/main.py 3969 &
python  ${path}/main.py 3970 &
python  ${path}/main.py 3971 &
python  ${path}/main.py 3972 &
python  ${path}/main.py 3973 &
python  ${path}/main.py 3974 &
python  ${path}/main.py 3975 &
python  ${path}/main.py 3976 &
python  ${path}/main.py 3977 &
python  ${path}/main.py 3978 &
python  ${path}/main.py 3979 &
python  ${path}/main.py 3980 &
python  ${path}/main.py 3981 &
python  ${path}/main.py 3982 &
python  ${path}/main.py 3983 &
python  ${path}/main.py 3984 &
python  ${path}/main.py 3985 &
python  ${path}/main.py 3986 &
python  ${path}/main.py 3987 &
python  ${path}/main.py 3988 &
python  ${path}/main.py 3989 &
python  ${path}/main.py 3990 &
python  ${path}/main.py 3991 &
python  ${path}/main.py 3992 &
python  ${path}/main.py 3993 &
python  ${path}/main.py 3994 &
python  ${path}/main.py 3995 &
python  ${path}/main.py 3996 &
python  ${path}/main.py 3997 &
python  ${path}/main.py 3998 &
python  ${path}/main.py 3999 &
python  ${path}/main.py 4000 &
python  ${path}/main.py 4001 &
python  ${path}/main.py 4002 &
python  ${path}/main.py 4003 &
python  ${path}/main.py 4004 &
python  ${path}/main.py 4005 &
python  ${path}/main.py 4006 &
python  ${path}/main.py 4007 &
python  ${path}/main.py 4008 &
python  ${path}/main.py 4009 &
python  ${path}/main.py 4010 &
python  ${path}/main.py 4011 &
python  ${path}/main.py 4012 &
python  ${path}/main.py 4013 &
python  ${path}/main.py 4014 &
python  ${path}/main.py 4015 &
python  ${path}/main.py 4016 &
python  ${path}/main.py 4017 &
python  ${path}/main.py 4018 &
python  ${path}/main.py 4019 &
python  ${path}/main.py 4020 &
python  ${path}/main.py 4021 &
python  ${path}/main.py 4022 &
python  ${path}/main.py 4023 &
python  ${path}/main.py 4024 &
python  ${path}/main.py 4025 &
python  ${path}/main.py 4026 &
python  ${path}/main.py 4027 &
python  ${path}/main.py 4028 &
python  ${path}/main.py 4029 &
python  ${path}/main.py 4030 &
python  ${path}/main.py 4031 &
wait
