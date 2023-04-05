#!/bin/bash -l
#SBATCH -J NN_129
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9216 &
python  ${path}/main.py 9217 &
python  ${path}/main.py 9218 &
python  ${path}/main.py 9219 &
python  ${path}/main.py 9220 &
python  ${path}/main.py 9221 &
python  ${path}/main.py 9222 &
python  ${path}/main.py 9223 &
python  ${path}/main.py 9224 &
python  ${path}/main.py 9225 &
python  ${path}/main.py 9226 &
python  ${path}/main.py 9227 &
python  ${path}/main.py 9228 &
python  ${path}/main.py 9229 &
python  ${path}/main.py 9230 &
python  ${path}/main.py 9231 &
python  ${path}/main.py 9232 &
python  ${path}/main.py 9233 &
python  ${path}/main.py 9234 &
python  ${path}/main.py 9235 &
python  ${path}/main.py 9236 &
python  ${path}/main.py 9237 &
python  ${path}/main.py 9238 &
python  ${path}/main.py 9239 &
python  ${path}/main.py 9240 &
python  ${path}/main.py 9241 &
python  ${path}/main.py 9242 &
python  ${path}/main.py 9243 &
python  ${path}/main.py 9244 &
python  ${path}/main.py 9245 &
python  ${path}/main.py 9246 &
python  ${path}/main.py 9247 &
python  ${path}/main.py 9248 &
python  ${path}/main.py 9249 &
python  ${path}/main.py 9250 &
python  ${path}/main.py 9251 &
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
