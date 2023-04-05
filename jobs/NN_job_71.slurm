#!/bin/bash -l
#SBATCH -J NN_71
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5040 &
python  ${path}/main.py 5041 &
python  ${path}/main.py 5042 &
python  ${path}/main.py 5043 &
python  ${path}/main.py 5044 &
python  ${path}/main.py 5045 &
python  ${path}/main.py 5046 &
python  ${path}/main.py 5047 &
python  ${path}/main.py 5048 &
python  ${path}/main.py 5049 &
python  ${path}/main.py 5050 &
python  ${path}/main.py 5051 &
python  ${path}/main.py 5052 &
python  ${path}/main.py 5053 &
python  ${path}/main.py 5054 &
python  ${path}/main.py 5055 &
python  ${path}/main.py 5056 &
python  ${path}/main.py 5057 &
python  ${path}/main.py 5058 &
python  ${path}/main.py 5059 &
python  ${path}/main.py 5060 &
python  ${path}/main.py 5061 &
python  ${path}/main.py 5062 &
python  ${path}/main.py 5063 &
python  ${path}/main.py 5064 &
python  ${path}/main.py 5065 &
python  ${path}/main.py 5066 &
python  ${path}/main.py 5067 &
python  ${path}/main.py 5068 &
python  ${path}/main.py 5069 &
python  ${path}/main.py 5070 &
python  ${path}/main.py 5071 &
python  ${path}/main.py 5072 &
python  ${path}/main.py 5073 &
python  ${path}/main.py 5074 &
python  ${path}/main.py 5075 &
python  ${path}/main.py 5076 &
python  ${path}/main.py 5077 &
python  ${path}/main.py 5078 &
python  ${path}/main.py 5079 &
python  ${path}/main.py 5080 &
python  ${path}/main.py 5081 &
python  ${path}/main.py 5082 &
python  ${path}/main.py 5083 &
python  ${path}/main.py 5084 &
python  ${path}/main.py 5085 &
python  ${path}/main.py 5086 &
python  ${path}/main.py 5087 &
python  ${path}/main.py 5088 &
python  ${path}/main.py 5089 &
python  ${path}/main.py 5090 &
python  ${path}/main.py 5091 &
python  ${path}/main.py 5092 &
python  ${path}/main.py 5093 &
python  ${path}/main.py 5094 &
python  ${path}/main.py 5095 &
python  ${path}/main.py 5096 &
python  ${path}/main.py 5097 &
python  ${path}/main.py 5098 &
python  ${path}/main.py 5099 &
python  ${path}/main.py 5100 &
python  ${path}/main.py 5101 &
python  ${path}/main.py 5102 &
python  ${path}/main.py 5103 &
python  ${path}/main.py 5104 &
python  ${path}/main.py 5105 &
python  ${path}/main.py 5106 &
python  ${path}/main.py 5107 &
python  ${path}/main.py 5108 &
python  ${path}/main.py 5109 &
python  ${path}/main.py 5110 &
python  ${path}/main.py 5111 &
wait
