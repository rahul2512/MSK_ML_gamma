#!/bin/bash -l
#SBATCH -J NN_131
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9360 &
python  ${path}/main.py 9361 &
python  ${path}/main.py 9362 &
python  ${path}/main.py 9363 &
python  ${path}/main.py 9364 &
python  ${path}/main.py 9365 &
python  ${path}/main.py 9366 &
python  ${path}/main.py 9367 &
python  ${path}/main.py 9368 &
python  ${path}/main.py 9369 &
python  ${path}/main.py 9370 &
python  ${path}/main.py 9371 &
python  ${path}/main.py 9372 &
python  ${path}/main.py 9373 &
python  ${path}/main.py 9374 &
python  ${path}/main.py 9375 &
python  ${path}/main.py 9376 &
python  ${path}/main.py 9377 &
python  ${path}/main.py 9378 &
python  ${path}/main.py 9379 &
python  ${path}/main.py 9380 &
python  ${path}/main.py 9381 &
python  ${path}/main.py 9382 &
python  ${path}/main.py 9383 &
python  ${path}/main.py 9384 &
python  ${path}/main.py 9385 &
python  ${path}/main.py 9386 &
python  ${path}/main.py 9387 &
python  ${path}/main.py 9388 &
python  ${path}/main.py 9389 &
python  ${path}/main.py 9390 &
python  ${path}/main.py 9391 &
python  ${path}/main.py 9392 &
python  ${path}/main.py 9393 &
python  ${path}/main.py 9394 &
python  ${path}/main.py 9395 &
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
