#!/bin/bash -l
#SBATCH -J RNN_144
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
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
