#!/bin/bash -l
#SBATCH -J NN_143
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
python  ${path}/main.py 10260 &
python  ${path}/main.py 10261 &
python  ${path}/main.py 10262 &
python  ${path}/main.py 10263 &
python  ${path}/main.py 10264 &
python  ${path}/main.py 10265 &
python  ${path}/main.py 10266 &
python  ${path}/main.py 10267 &
python  ${path}/main.py 10268 &
python  ${path}/main.py 10269 &
python  ${path}/main.py 10270 &
python  ${path}/main.py 10271 &
python  ${path}/main.py 10272 &
python  ${path}/main.py 10273 &
python  ${path}/main.py 10274 &
python  ${path}/main.py 10275 &
python  ${path}/main.py 10276 &
python  ${path}/main.py 10277 &
python  ${path}/main.py 10278 &
python  ${path}/main.py 10279 &
python  ${path}/main.py 10280 &
python  ${path}/main.py 10281 &
python  ${path}/main.py 10282 &
python  ${path}/main.py 10283 &
python  ${path}/main.py 10284 &
python  ${path}/main.py 10285 &
python  ${path}/main.py 10286 &
python  ${path}/main.py 10287 &
python  ${path}/main.py 10288 &
python  ${path}/main.py 10289 &
python  ${path}/main.py 10290 &
python  ${path}/main.py 10291 &
python  ${path}/main.py 10292 &
python  ${path}/main.py 10293 &
python  ${path}/main.py 10294 &
python  ${path}/main.py 10295 &
wait
