#!/bin/bash -l
#SBATCH -J NN_83
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5904 &
python  ${path}/main.py 5905 &
python  ${path}/main.py 5906 &
python  ${path}/main.py 5907 &
python  ${path}/main.py 5908 &
python  ${path}/main.py 5909 &
python  ${path}/main.py 5910 &
python  ${path}/main.py 5911 &
python  ${path}/main.py 5912 &
python  ${path}/main.py 5913 &
python  ${path}/main.py 5914 &
python  ${path}/main.py 5915 &
python  ${path}/main.py 5916 &
python  ${path}/main.py 5917 &
python  ${path}/main.py 5918 &
python  ${path}/main.py 5919 &
python  ${path}/main.py 5920 &
python  ${path}/main.py 5921 &
python  ${path}/main.py 5922 &
python  ${path}/main.py 5923 &
python  ${path}/main.py 5924 &
python  ${path}/main.py 5925 &
python  ${path}/main.py 5926 &
python  ${path}/main.py 5927 &
python  ${path}/main.py 5928 &
python  ${path}/main.py 5929 &
python  ${path}/main.py 5930 &
python  ${path}/main.py 5931 &
python  ${path}/main.py 5932 &
python  ${path}/main.py 5933 &
python  ${path}/main.py 5934 &
python  ${path}/main.py 5935 &
python  ${path}/main.py 5936 &
python  ${path}/main.py 5937 &
python  ${path}/main.py 5938 &
python  ${path}/main.py 5939 &
python  ${path}/main.py 5940 &
python  ${path}/main.py 5941 &
python  ${path}/main.py 5942 &
python  ${path}/main.py 5943 &
python  ${path}/main.py 5944 &
python  ${path}/main.py 5945 &
python  ${path}/main.py 5946 &
python  ${path}/main.py 5947 &
python  ${path}/main.py 5948 &
python  ${path}/main.py 5949 &
python  ${path}/main.py 5950 &
python  ${path}/main.py 5951 &
python  ${path}/main.py 5952 &
python  ${path}/main.py 5953 &
python  ${path}/main.py 5954 &
python  ${path}/main.py 5955 &
python  ${path}/main.py 5956 &
python  ${path}/main.py 5957 &
python  ${path}/main.py 5958 &
python  ${path}/main.py 5959 &
python  ${path}/main.py 5960 &
python  ${path}/main.py 5961 &
python  ${path}/main.py 5962 &
python  ${path}/main.py 5963 &
python  ${path}/main.py 5964 &
python  ${path}/main.py 5965 &
python  ${path}/main.py 5966 &
python  ${path}/main.py 5967 &
python  ${path}/main.py 5968 &
python  ${path}/main.py 5969 &
python  ${path}/main.py 5970 &
python  ${path}/main.py 5971 &
python  ${path}/main.py 5972 &
python  ${path}/main.py 5973 &
python  ${path}/main.py 5974 &
python  ${path}/main.py 5975 &
wait
