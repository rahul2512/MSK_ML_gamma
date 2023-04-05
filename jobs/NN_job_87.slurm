#!/bin/bash -l
#SBATCH -J NN_87
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6192 &
python  ${path}/main.py 6193 &
python  ${path}/main.py 6194 &
python  ${path}/main.py 6195 &
python  ${path}/main.py 6196 &
python  ${path}/main.py 6197 &
python  ${path}/main.py 6198 &
python  ${path}/main.py 6199 &
python  ${path}/main.py 6200 &
python  ${path}/main.py 6201 &
python  ${path}/main.py 6202 &
python  ${path}/main.py 6203 &
python  ${path}/main.py 6204 &
python  ${path}/main.py 6205 &
python  ${path}/main.py 6206 &
python  ${path}/main.py 6207 &
python  ${path}/main.py 6208 &
python  ${path}/main.py 6209 &
python  ${path}/main.py 6210 &
python  ${path}/main.py 6211 &
python  ${path}/main.py 6212 &
python  ${path}/main.py 6213 &
python  ${path}/main.py 6214 &
python  ${path}/main.py 6215 &
python  ${path}/main.py 6216 &
python  ${path}/main.py 6217 &
python  ${path}/main.py 6218 &
python  ${path}/main.py 6219 &
python  ${path}/main.py 6220 &
python  ${path}/main.py 6221 &
python  ${path}/main.py 6222 &
python  ${path}/main.py 6223 &
python  ${path}/main.py 6224 &
python  ${path}/main.py 6225 &
python  ${path}/main.py 6226 &
python  ${path}/main.py 6227 &
python  ${path}/main.py 6228 &
python  ${path}/main.py 6229 &
python  ${path}/main.py 6230 &
python  ${path}/main.py 6231 &
python  ${path}/main.py 6232 &
python  ${path}/main.py 6233 &
python  ${path}/main.py 6234 &
python  ${path}/main.py 6235 &
python  ${path}/main.py 6236 &
python  ${path}/main.py 6237 &
python  ${path}/main.py 6238 &
python  ${path}/main.py 6239 &
python  ${path}/main.py 6240 &
python  ${path}/main.py 6241 &
python  ${path}/main.py 6242 &
python  ${path}/main.py 6243 &
python  ${path}/main.py 6244 &
python  ${path}/main.py 6245 &
python  ${path}/main.py 6246 &
python  ${path}/main.py 6247 &
python  ${path}/main.py 6248 &
python  ${path}/main.py 6249 &
python  ${path}/main.py 6250 &
python  ${path}/main.py 6251 &
python  ${path}/main.py 6252 &
python  ${path}/main.py 6253 &
python  ${path}/main.py 6254 &
python  ${path}/main.py 6255 &
python  ${path}/main.py 6256 &
python  ${path}/main.py 6257 &
python  ${path}/main.py 6258 &
python  ${path}/main.py 6259 &
python  ${path}/main.py 6260 &
python  ${path}/main.py 6261 &
python  ${path}/main.py 6262 &
python  ${path}/main.py 6263 &
wait
