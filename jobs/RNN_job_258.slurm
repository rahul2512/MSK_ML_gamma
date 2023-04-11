#!/bin/bash -l
#SBATCH -J RNN_258
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9252 &
python  ${path}/main.py 9253 &
python  ${path}/main.py 9254 &
python  ${path}/main.py 9255 &
python  ${path}/main.py 9256 &
python  ${path}/main.py 9257 &
python  ${path}/main.py 9258 &
python  ${path}/main.py 9259 &
python  ${path}/main.py 9260 &
python  ${path}/main.py 9261 &
python  ${path}/main.py 9262 &
python  ${path}/main.py 9263 &
python  ${path}/main.py 9264 &
python  ${path}/main.py 9265 &
python  ${path}/main.py 9266 &
python  ${path}/main.py 9267 &
python  ${path}/main.py 9268 &
python  ${path}/main.py 9269 &
python  ${path}/main.py 9270 &
python  ${path}/main.py 9271 &
python  ${path}/main.py 9272 &
python  ${path}/main.py 9273 &
python  ${path}/main.py 9274 &
python  ${path}/main.py 9275 &
python  ${path}/main.py 9276 &
python  ${path}/main.py 9277 &
python  ${path}/main.py 9278 &
python  ${path}/main.py 9279 &
python  ${path}/main.py 9280 &
python  ${path}/main.py 9281 &
python  ${path}/main.py 9282 &
python  ${path}/main.py 9283 &
python  ${path}/main.py 9284 &
python  ${path}/main.py 9285 &
python  ${path}/main.py 9286 &
python  ${path}/main.py 9287 &
wait
