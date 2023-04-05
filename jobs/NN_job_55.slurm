#!/bin/bash -l
#SBATCH -J NN_55
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3888 &
python  ${path}/main.py 3889 &
python  ${path}/main.py 3890 &
python  ${path}/main.py 3891 &
python  ${path}/main.py 3892 &
python  ${path}/main.py 3893 &
python  ${path}/main.py 3894 &
python  ${path}/main.py 3895 &
python  ${path}/main.py 3896 &
python  ${path}/main.py 3897 &
python  ${path}/main.py 3898 &
python  ${path}/main.py 3899 &
python  ${path}/main.py 3900 &
python  ${path}/main.py 3901 &
python  ${path}/main.py 3902 &
python  ${path}/main.py 3903 &
python  ${path}/main.py 3904 &
python  ${path}/main.py 3905 &
python  ${path}/main.py 3906 &
python  ${path}/main.py 3907 &
python  ${path}/main.py 3908 &
python  ${path}/main.py 3909 &
python  ${path}/main.py 3910 &
python  ${path}/main.py 3911 &
python  ${path}/main.py 3912 &
python  ${path}/main.py 3913 &
python  ${path}/main.py 3914 &
python  ${path}/main.py 3915 &
python  ${path}/main.py 3916 &
python  ${path}/main.py 3917 &
python  ${path}/main.py 3918 &
python  ${path}/main.py 3919 &
python  ${path}/main.py 3920 &
python  ${path}/main.py 3921 &
python  ${path}/main.py 3922 &
python  ${path}/main.py 3923 &
python  ${path}/main.py 3924 &
python  ${path}/main.py 3925 &
python  ${path}/main.py 3926 &
python  ${path}/main.py 3927 &
python  ${path}/main.py 3928 &
python  ${path}/main.py 3929 &
python  ${path}/main.py 3930 &
python  ${path}/main.py 3931 &
python  ${path}/main.py 3932 &
python  ${path}/main.py 3933 &
python  ${path}/main.py 3934 &
python  ${path}/main.py 3935 &
python  ${path}/main.py 3936 &
python  ${path}/main.py 3937 &
python  ${path}/main.py 3938 &
python  ${path}/main.py 3939 &
python  ${path}/main.py 3940 &
python  ${path}/main.py 3941 &
python  ${path}/main.py 3942 &
python  ${path}/main.py 3943 &
python  ${path}/main.py 3944 &
python  ${path}/main.py 3945 &
python  ${path}/main.py 3946 &
python  ${path}/main.py 3947 &
python  ${path}/main.py 3948 &
python  ${path}/main.py 3949 &
python  ${path}/main.py 3950 &
python  ${path}/main.py 3951 &
python  ${path}/main.py 3952 &
python  ${path}/main.py 3953 &
python  ${path}/main.py 3954 &
python  ${path}/main.py 3955 &
python  ${path}/main.py 3956 &
python  ${path}/main.py 3957 &
python  ${path}/main.py 3958 &
python  ${path}/main.py 3959 &
wait
