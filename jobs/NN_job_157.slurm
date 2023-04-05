#!/bin/bash -l
#SBATCH -J NN_157
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 11232 &
python  ${path}/main.py 11233 &
python  ${path}/main.py 11234 &
python  ${path}/main.py 11235 &
python  ${path}/main.py 11236 &
python  ${path}/main.py 11237 &
python  ${path}/main.py 11238 &
python  ${path}/main.py 11239 &
python  ${path}/main.py 11240 &
python  ${path}/main.py 11241 &
python  ${path}/main.py 11242 &
python  ${path}/main.py 11243 &
python  ${path}/main.py 11244 &
python  ${path}/main.py 11245 &
python  ${path}/main.py 11246 &
python  ${path}/main.py 11247 &
python  ${path}/main.py 11248 &
python  ${path}/main.py 11249 &
python  ${path}/main.py 11250 &
python  ${path}/main.py 11251 &
python  ${path}/main.py 11252 &
python  ${path}/main.py 11253 &
python  ${path}/main.py 11254 &
python  ${path}/main.py 11255 &
python  ${path}/main.py 11256 &
python  ${path}/main.py 11257 &
python  ${path}/main.py 11258 &
python  ${path}/main.py 11259 &
python  ${path}/main.py 11260 &
python  ${path}/main.py 11261 &
python  ${path}/main.py 11262 &
python  ${path}/main.py 11263 &
python  ${path}/main.py 11264 &
python  ${path}/main.py 11265 &
python  ${path}/main.py 11266 &
python  ${path}/main.py 11267 &
python  ${path}/main.py 11268 &
python  ${path}/main.py 11269 &
python  ${path}/main.py 11270 &
python  ${path}/main.py 11271 &
python  ${path}/main.py 11272 &
python  ${path}/main.py 11273 &
python  ${path}/main.py 11274 &
python  ${path}/main.py 11275 &
python  ${path}/main.py 11276 &
python  ${path}/main.py 11277 &
python  ${path}/main.py 11278 &
python  ${path}/main.py 11279 &
python  ${path}/main.py 11280 &
python  ${path}/main.py 11281 &
python  ${path}/main.py 11282 &
python  ${path}/main.py 11283 &
python  ${path}/main.py 11284 &
python  ${path}/main.py 11285 &
python  ${path}/main.py 11286 &
python  ${path}/main.py 11287 &
python  ${path}/main.py 11288 &
python  ${path}/main.py 11289 &
python  ${path}/main.py 11290 &
python  ${path}/main.py 11291 &
python  ${path}/main.py 11292 &
python  ${path}/main.py 11293 &
python  ${path}/main.py 11294 &
python  ${path}/main.py 11295 &
python  ${path}/main.py 11296 &
python  ${path}/main.py 11297 &
python  ${path}/main.py 11298 &
python  ${path}/main.py 11299 &
python  ${path}/main.py 11300 &
python  ${path}/main.py 11301 &
python  ${path}/main.py 11302 &
python  ${path}/main.py 11303 &
wait
