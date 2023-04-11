#!/bin/bash -l
#SBATCH -J RNN_283
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10152 &
python  ${path}/main.py 10153 &
python  ${path}/main.py 10154 &
python  ${path}/main.py 10155 &
python  ${path}/main.py 10156 &
python  ${path}/main.py 10157 &
python  ${path}/main.py 10158 &
python  ${path}/main.py 10159 &
python  ${path}/main.py 10160 &
python  ${path}/main.py 10161 &
python  ${path}/main.py 10162 &
python  ${path}/main.py 10163 &
python  ${path}/main.py 10164 &
python  ${path}/main.py 10165 &
python  ${path}/main.py 10166 &
python  ${path}/main.py 10167 &
python  ${path}/main.py 10168 &
python  ${path}/main.py 10169 &
python  ${path}/main.py 10170 &
python  ${path}/main.py 10171 &
python  ${path}/main.py 10172 &
python  ${path}/main.py 10173 &
python  ${path}/main.py 10174 &
python  ${path}/main.py 10175 &
python  ${path}/main.py 10176 &
python  ${path}/main.py 10177 &
python  ${path}/main.py 10178 &
python  ${path}/main.py 10179 &
python  ${path}/main.py 10180 &
python  ${path}/main.py 10181 &
python  ${path}/main.py 10182 &
python  ${path}/main.py 10183 &
python  ${path}/main.py 10184 &
python  ${path}/main.py 10185 &
python  ${path}/main.py 10186 &
python  ${path}/main.py 10187 &
wait
