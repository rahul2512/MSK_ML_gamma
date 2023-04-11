#!/bin/bash -l
#SBATCH -J RNN_262
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9396 &
python  ${path}/main.py 9397 &
python  ${path}/main.py 9398 &
python  ${path}/main.py 9399 &
python  ${path}/main.py 9400 &
python  ${path}/main.py 9401 &
python  ${path}/main.py 9402 &
python  ${path}/main.py 9403 &
python  ${path}/main.py 9404 &
python  ${path}/main.py 9405 &
python  ${path}/main.py 9406 &
python  ${path}/main.py 9407 &
python  ${path}/main.py 9408 &
python  ${path}/main.py 9409 &
python  ${path}/main.py 9410 &
python  ${path}/main.py 9411 &
python  ${path}/main.py 9412 &
python  ${path}/main.py 9413 &
python  ${path}/main.py 9414 &
python  ${path}/main.py 9415 &
python  ${path}/main.py 9416 &
python  ${path}/main.py 9417 &
python  ${path}/main.py 9418 &
python  ${path}/main.py 9419 &
python  ${path}/main.py 9420 &
python  ${path}/main.py 9421 &
python  ${path}/main.py 9422 &
python  ${path}/main.py 9423 &
python  ${path}/main.py 9424 &
python  ${path}/main.py 9425 &
python  ${path}/main.py 9426 &
python  ${path}/main.py 9427 &
python  ${path}/main.py 9428 &
python  ${path}/main.py 9429 &
python  ${path}/main.py 9430 &
python  ${path}/main.py 9431 &
wait
