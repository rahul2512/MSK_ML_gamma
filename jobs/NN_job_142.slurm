#!/bin/bash -l
#SBATCH -J NN_142
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
python  ${path}/main.py 10188 &
python  ${path}/main.py 10189 &
python  ${path}/main.py 10190 &
python  ${path}/main.py 10191 &
python  ${path}/main.py 10192 &
python  ${path}/main.py 10193 &
python  ${path}/main.py 10194 &
python  ${path}/main.py 10195 &
python  ${path}/main.py 10196 &
python  ${path}/main.py 10197 &
python  ${path}/main.py 10198 &
python  ${path}/main.py 10199 &
python  ${path}/main.py 10200 &
python  ${path}/main.py 10201 &
python  ${path}/main.py 10202 &
python  ${path}/main.py 10203 &
python  ${path}/main.py 10204 &
python  ${path}/main.py 10205 &
python  ${path}/main.py 10206 &
python  ${path}/main.py 10207 &
python  ${path}/main.py 10208 &
python  ${path}/main.py 10209 &
python  ${path}/main.py 10210 &
python  ${path}/main.py 10211 &
python  ${path}/main.py 10212 &
python  ${path}/main.py 10213 &
python  ${path}/main.py 10214 &
python  ${path}/main.py 10215 &
python  ${path}/main.py 10216 &
python  ${path}/main.py 10217 &
python  ${path}/main.py 10218 &
python  ${path}/main.py 10219 &
python  ${path}/main.py 10220 &
python  ${path}/main.py 10221 &
python  ${path}/main.py 10222 &
python  ${path}/main.py 10223 &
wait
