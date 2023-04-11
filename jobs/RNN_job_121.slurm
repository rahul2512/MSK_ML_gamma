#!/bin/bash -l
#SBATCH -J RNN_121
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4320 &
python  ${path}/main.py 4321 &
python  ${path}/main.py 4322 &
python  ${path}/main.py 4323 &
python  ${path}/main.py 4324 &
python  ${path}/main.py 4325 &
python  ${path}/main.py 4326 &
python  ${path}/main.py 4327 &
python  ${path}/main.py 4328 &
python  ${path}/main.py 4329 &
python  ${path}/main.py 4330 &
python  ${path}/main.py 4331 &
python  ${path}/main.py 4332 &
python  ${path}/main.py 4333 &
python  ${path}/main.py 4334 &
python  ${path}/main.py 4335 &
python  ${path}/main.py 4336 &
python  ${path}/main.py 4337 &
python  ${path}/main.py 4338 &
python  ${path}/main.py 4339 &
python  ${path}/main.py 4340 &
python  ${path}/main.py 4341 &
python  ${path}/main.py 4342 &
python  ${path}/main.py 4343 &
python  ${path}/main.py 4344 &
python  ${path}/main.py 4345 &
python  ${path}/main.py 4346 &
python  ${path}/main.py 4347 &
python  ${path}/main.py 4348 &
python  ${path}/main.py 4349 &
python  ${path}/main.py 4350 &
python  ${path}/main.py 4351 &
python  ${path}/main.py 4352 &
python  ${path}/main.py 4353 &
python  ${path}/main.py 4354 &
python  ${path}/main.py 4355 &
wait
