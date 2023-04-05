#!/bin/bash -l
#SBATCH -J NN_72
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5112 &
python  ${path}/main.py 5113 &
python  ${path}/main.py 5114 &
python  ${path}/main.py 5115 &
python  ${path}/main.py 5116 &
python  ${path}/main.py 5117 &
python  ${path}/main.py 5118 &
python  ${path}/main.py 5119 &
python  ${path}/main.py 5120 &
python  ${path}/main.py 5121 &
python  ${path}/main.py 5122 &
python  ${path}/main.py 5123 &
python  ${path}/main.py 5124 &
python  ${path}/main.py 5125 &
python  ${path}/main.py 5126 &
python  ${path}/main.py 5127 &
python  ${path}/main.py 5128 &
python  ${path}/main.py 5129 &
python  ${path}/main.py 5130 &
python  ${path}/main.py 5131 &
python  ${path}/main.py 5132 &
python  ${path}/main.py 5133 &
python  ${path}/main.py 5134 &
python  ${path}/main.py 5135 &
python  ${path}/main.py 5136 &
python  ${path}/main.py 5137 &
python  ${path}/main.py 5138 &
python  ${path}/main.py 5139 &
python  ${path}/main.py 5140 &
python  ${path}/main.py 5141 &
python  ${path}/main.py 5142 &
python  ${path}/main.py 5143 &
python  ${path}/main.py 5144 &
python  ${path}/main.py 5145 &
python  ${path}/main.py 5146 &
python  ${path}/main.py 5147 &
python  ${path}/main.py 5148 &
python  ${path}/main.py 5149 &
python  ${path}/main.py 5150 &
python  ${path}/main.py 5151 &
python  ${path}/main.py 5152 &
python  ${path}/main.py 5153 &
python  ${path}/main.py 5154 &
python  ${path}/main.py 5155 &
python  ${path}/main.py 5156 &
python  ${path}/main.py 5157 &
python  ${path}/main.py 5158 &
python  ${path}/main.py 5159 &
python  ${path}/main.py 5160 &
python  ${path}/main.py 5161 &
python  ${path}/main.py 5162 &
python  ${path}/main.py 5163 &
python  ${path}/main.py 5164 &
python  ${path}/main.py 5165 &
python  ${path}/main.py 5166 &
python  ${path}/main.py 5167 &
python  ${path}/main.py 5168 &
python  ${path}/main.py 5169 &
python  ${path}/main.py 5170 &
python  ${path}/main.py 5171 &
python  ${path}/main.py 5172 &
python  ${path}/main.py 5173 &
python  ${path}/main.py 5174 &
python  ${path}/main.py 5175 &
python  ${path}/main.py 5176 &
python  ${path}/main.py 5177 &
python  ${path}/main.py 5178 &
python  ${path}/main.py 5179 &
python  ${path}/main.py 5180 &
python  ${path}/main.py 5181 &
python  ${path}/main.py 5182 &
python  ${path}/main.py 5183 &
wait
