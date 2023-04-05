#!/bin/bash -l
#SBATCH -J NN_88
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6264 &
python  ${path}/main.py 6265 &
python  ${path}/main.py 6266 &
python  ${path}/main.py 6267 &
python  ${path}/main.py 6268 &
python  ${path}/main.py 6269 &
python  ${path}/main.py 6270 &
python  ${path}/main.py 6271 &
python  ${path}/main.py 6272 &
python  ${path}/main.py 6273 &
python  ${path}/main.py 6274 &
python  ${path}/main.py 6275 &
python  ${path}/main.py 6276 &
python  ${path}/main.py 6277 &
python  ${path}/main.py 6278 &
python  ${path}/main.py 6279 &
python  ${path}/main.py 6280 &
python  ${path}/main.py 6281 &
python  ${path}/main.py 6282 &
python  ${path}/main.py 6283 &
python  ${path}/main.py 6284 &
python  ${path}/main.py 6285 &
python  ${path}/main.py 6286 &
python  ${path}/main.py 6287 &
python  ${path}/main.py 6288 &
python  ${path}/main.py 6289 &
python  ${path}/main.py 6290 &
python  ${path}/main.py 6291 &
python  ${path}/main.py 6292 &
python  ${path}/main.py 6293 &
python  ${path}/main.py 6294 &
python  ${path}/main.py 6295 &
python  ${path}/main.py 6296 &
python  ${path}/main.py 6297 &
python  ${path}/main.py 6298 &
python  ${path}/main.py 6299 &
python  ${path}/main.py 6300 &
python  ${path}/main.py 6301 &
python  ${path}/main.py 6302 &
python  ${path}/main.py 6303 &
python  ${path}/main.py 6304 &
python  ${path}/main.py 6305 &
python  ${path}/main.py 6306 &
python  ${path}/main.py 6307 &
python  ${path}/main.py 6308 &
python  ${path}/main.py 6309 &
python  ${path}/main.py 6310 &
python  ${path}/main.py 6311 &
python  ${path}/main.py 6312 &
python  ${path}/main.py 6313 &
python  ${path}/main.py 6314 &
python  ${path}/main.py 6315 &
python  ${path}/main.py 6316 &
python  ${path}/main.py 6317 &
python  ${path}/main.py 6318 &
python  ${path}/main.py 6319 &
python  ${path}/main.py 6320 &
python  ${path}/main.py 6321 &
python  ${path}/main.py 6322 &
python  ${path}/main.py 6323 &
python  ${path}/main.py 6324 &
python  ${path}/main.py 6325 &
python  ${path}/main.py 6326 &
python  ${path}/main.py 6327 &
python  ${path}/main.py 6328 &
python  ${path}/main.py 6329 &
python  ${path}/main.py 6330 &
python  ${path}/main.py 6331 &
python  ${path}/main.py 6332 &
python  ${path}/main.py 6333 &
python  ${path}/main.py 6334 &
python  ${path}/main.py 6335 &
wait
