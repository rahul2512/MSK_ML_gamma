#!/bin/bash -l
#SBATCH -J RNN_138
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4932 &
python  ${path}/main.py 4933 &
python  ${path}/main.py 4934 &
python  ${path}/main.py 4935 &
python  ${path}/main.py 4936 &
python  ${path}/main.py 4937 &
python  ${path}/main.py 4938 &
python  ${path}/main.py 4939 &
python  ${path}/main.py 4940 &
python  ${path}/main.py 4941 &
python  ${path}/main.py 4942 &
python  ${path}/main.py 4943 &
python  ${path}/main.py 4944 &
python  ${path}/main.py 4945 &
python  ${path}/main.py 4946 &
python  ${path}/main.py 4947 &
python  ${path}/main.py 4948 &
python  ${path}/main.py 4949 &
python  ${path}/main.py 4950 &
python  ${path}/main.py 4951 &
python  ${path}/main.py 4952 &
python  ${path}/main.py 4953 &
python  ${path}/main.py 4954 &
python  ${path}/main.py 4955 &
python  ${path}/main.py 4956 &
python  ${path}/main.py 4957 &
python  ${path}/main.py 4958 &
python  ${path}/main.py 4959 &
python  ${path}/main.py 4960 &
python  ${path}/main.py 4961 &
python  ${path}/main.py 4962 &
python  ${path}/main.py 4963 &
python  ${path}/main.py 4964 &
python  ${path}/main.py 4965 &
python  ${path}/main.py 4966 &
python  ${path}/main.py 4967 &
wait
