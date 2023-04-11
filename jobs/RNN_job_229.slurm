#!/bin/bash -l
#SBATCH -J RNN_229
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8208 &
python  ${path}/main.py 8209 &
python  ${path}/main.py 8210 &
python  ${path}/main.py 8211 &
python  ${path}/main.py 8212 &
python  ${path}/main.py 8213 &
python  ${path}/main.py 8214 &
python  ${path}/main.py 8215 &
python  ${path}/main.py 8216 &
python  ${path}/main.py 8217 &
python  ${path}/main.py 8218 &
python  ${path}/main.py 8219 &
python  ${path}/main.py 8220 &
python  ${path}/main.py 8221 &
python  ${path}/main.py 8222 &
python  ${path}/main.py 8223 &
python  ${path}/main.py 8224 &
python  ${path}/main.py 8225 &
python  ${path}/main.py 8226 &
python  ${path}/main.py 8227 &
python  ${path}/main.py 8228 &
python  ${path}/main.py 8229 &
python  ${path}/main.py 8230 &
python  ${path}/main.py 8231 &
python  ${path}/main.py 8232 &
python  ${path}/main.py 8233 &
python  ${path}/main.py 8234 &
python  ${path}/main.py 8235 &
python  ${path}/main.py 8236 &
python  ${path}/main.py 8237 &
python  ${path}/main.py 8238 &
python  ${path}/main.py 8239 &
python  ${path}/main.py 8240 &
python  ${path}/main.py 8241 &
python  ${path}/main.py 8242 &
python  ${path}/main.py 8243 &
wait
