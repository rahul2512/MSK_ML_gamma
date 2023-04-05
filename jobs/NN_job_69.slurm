#!/bin/bash -l
#SBATCH -J NN_69
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4896 &
python  ${path}/main.py 4897 &
python  ${path}/main.py 4898 &
python  ${path}/main.py 4899 &
python  ${path}/main.py 4900 &
python  ${path}/main.py 4901 &
python  ${path}/main.py 4902 &
python  ${path}/main.py 4903 &
python  ${path}/main.py 4904 &
python  ${path}/main.py 4905 &
python  ${path}/main.py 4906 &
python  ${path}/main.py 4907 &
python  ${path}/main.py 4908 &
python  ${path}/main.py 4909 &
python  ${path}/main.py 4910 &
python  ${path}/main.py 4911 &
python  ${path}/main.py 4912 &
python  ${path}/main.py 4913 &
python  ${path}/main.py 4914 &
python  ${path}/main.py 4915 &
python  ${path}/main.py 4916 &
python  ${path}/main.py 4917 &
python  ${path}/main.py 4918 &
python  ${path}/main.py 4919 &
python  ${path}/main.py 4920 &
python  ${path}/main.py 4921 &
python  ${path}/main.py 4922 &
python  ${path}/main.py 4923 &
python  ${path}/main.py 4924 &
python  ${path}/main.py 4925 &
python  ${path}/main.py 4926 &
python  ${path}/main.py 4927 &
python  ${path}/main.py 4928 &
python  ${path}/main.py 4929 &
python  ${path}/main.py 4930 &
python  ${path}/main.py 4931 &
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
