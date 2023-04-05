#!/bin/bash -l
#SBATCH -J NN_101
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7200 &
python  ${path}/main.py 7201 &
python  ${path}/main.py 7202 &
python  ${path}/main.py 7203 &
python  ${path}/main.py 7204 &
python  ${path}/main.py 7205 &
python  ${path}/main.py 7206 &
python  ${path}/main.py 7207 &
python  ${path}/main.py 7208 &
python  ${path}/main.py 7209 &
python  ${path}/main.py 7210 &
python  ${path}/main.py 7211 &
python  ${path}/main.py 7212 &
python  ${path}/main.py 7213 &
python  ${path}/main.py 7214 &
python  ${path}/main.py 7215 &
python  ${path}/main.py 7216 &
python  ${path}/main.py 7217 &
python  ${path}/main.py 7218 &
python  ${path}/main.py 7219 &
python  ${path}/main.py 7220 &
python  ${path}/main.py 7221 &
python  ${path}/main.py 7222 &
python  ${path}/main.py 7223 &
python  ${path}/main.py 7224 &
python  ${path}/main.py 7225 &
python  ${path}/main.py 7226 &
python  ${path}/main.py 7227 &
python  ${path}/main.py 7228 &
python  ${path}/main.py 7229 &
python  ${path}/main.py 7230 &
python  ${path}/main.py 7231 &
python  ${path}/main.py 7232 &
python  ${path}/main.py 7233 &
python  ${path}/main.py 7234 &
python  ${path}/main.py 7235 &
python  ${path}/main.py 7236 &
python  ${path}/main.py 7237 &
python  ${path}/main.py 7238 &
python  ${path}/main.py 7239 &
python  ${path}/main.py 7240 &
python  ${path}/main.py 7241 &
python  ${path}/main.py 7242 &
python  ${path}/main.py 7243 &
python  ${path}/main.py 7244 &
python  ${path}/main.py 7245 &
python  ${path}/main.py 7246 &
python  ${path}/main.py 7247 &
python  ${path}/main.py 7248 &
python  ${path}/main.py 7249 &
python  ${path}/main.py 7250 &
python  ${path}/main.py 7251 &
python  ${path}/main.py 7252 &
python  ${path}/main.py 7253 &
python  ${path}/main.py 7254 &
python  ${path}/main.py 7255 &
python  ${path}/main.py 7256 &
python  ${path}/main.py 7257 &
python  ${path}/main.py 7258 &
python  ${path}/main.py 7259 &
python  ${path}/main.py 7260 &
python  ${path}/main.py 7261 &
python  ${path}/main.py 7262 &
python  ${path}/main.py 7263 &
python  ${path}/main.py 7264 &
python  ${path}/main.py 7265 &
python  ${path}/main.py 7266 &
python  ${path}/main.py 7267 &
python  ${path}/main.py 7268 &
python  ${path}/main.py 7269 &
python  ${path}/main.py 7270 &
python  ${path}/main.py 7271 &
wait
