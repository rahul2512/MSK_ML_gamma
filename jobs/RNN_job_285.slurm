#!/bin/bash -l
#SBATCH -J RNN_285
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10224 &
python  ${path}/main.py 10225 &
python  ${path}/main.py 10226 &
python  ${path}/main.py 10227 &
python  ${path}/main.py 10228 &
python  ${path}/main.py 10229 &
python  ${path}/main.py 10230 &
python  ${path}/main.py 10231 &
python  ${path}/main.py 10232 &
python  ${path}/main.py 10233 &
python  ${path}/main.py 10234 &
python  ${path}/main.py 10235 &
python  ${path}/main.py 10236 &
python  ${path}/main.py 10237 &
python  ${path}/main.py 10238 &
python  ${path}/main.py 10239 &
python  ${path}/main.py 10240 &
python  ${path}/main.py 10241 &
python  ${path}/main.py 10242 &
python  ${path}/main.py 10243 &
python  ${path}/main.py 10244 &
python  ${path}/main.py 10245 &
python  ${path}/main.py 10246 &
python  ${path}/main.py 10247 &
python  ${path}/main.py 10248 &
python  ${path}/main.py 10249 &
python  ${path}/main.py 10250 &
python  ${path}/main.py 10251 &
python  ${path}/main.py 10252 &
python  ${path}/main.py 10253 &
python  ${path}/main.py 10254 &
python  ${path}/main.py 10255 &
python  ${path}/main.py 10256 &
python  ${path}/main.py 10257 &
python  ${path}/main.py 10258 &
python  ${path}/main.py 10259 &
wait
