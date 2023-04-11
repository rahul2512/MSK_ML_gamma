#!/bin/bash -l
#SBATCH -J RNN_5
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 144 &
python  ${path}/main.py 145 &
python  ${path}/main.py 146 &
python  ${path}/main.py 147 &
python  ${path}/main.py 148 &
python  ${path}/main.py 149 &
python  ${path}/main.py 150 &
python  ${path}/main.py 151 &
python  ${path}/main.py 152 &
python  ${path}/main.py 153 &
python  ${path}/main.py 154 &
python  ${path}/main.py 155 &
python  ${path}/main.py 156 &
python  ${path}/main.py 157 &
python  ${path}/main.py 158 &
python  ${path}/main.py 159 &
python  ${path}/main.py 160 &
python  ${path}/main.py 161 &
python  ${path}/main.py 162 &
python  ${path}/main.py 163 &
python  ${path}/main.py 164 &
python  ${path}/main.py 165 &
python  ${path}/main.py 166 &
python  ${path}/main.py 167 &
python  ${path}/main.py 168 &
python  ${path}/main.py 169 &
python  ${path}/main.py 170 &
python  ${path}/main.py 171 &
python  ${path}/main.py 172 &
python  ${path}/main.py 173 &
python  ${path}/main.py 174 &
python  ${path}/main.py 175 &
python  ${path}/main.py 176 &
python  ${path}/main.py 177 &
python  ${path}/main.py 178 &
python  ${path}/main.py 179 &
wait
