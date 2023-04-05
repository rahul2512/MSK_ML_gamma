#!/bin/bash -l
#SBATCH -J NN_61
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
python  ${path}/main.py 4356 &
python  ${path}/main.py 4357 &
python  ${path}/main.py 4358 &
python  ${path}/main.py 4359 &
python  ${path}/main.py 4360 &
python  ${path}/main.py 4361 &
python  ${path}/main.py 4362 &
python  ${path}/main.py 4363 &
python  ${path}/main.py 4364 &
python  ${path}/main.py 4365 &
python  ${path}/main.py 4366 &
python  ${path}/main.py 4367 &
python  ${path}/main.py 4368 &
python  ${path}/main.py 4369 &
python  ${path}/main.py 4370 &
python  ${path}/main.py 4371 &
python  ${path}/main.py 4372 &
python  ${path}/main.py 4373 &
python  ${path}/main.py 4374 &
python  ${path}/main.py 4375 &
python  ${path}/main.py 4376 &
python  ${path}/main.py 4377 &
python  ${path}/main.py 4378 &
python  ${path}/main.py 4379 &
python  ${path}/main.py 4380 &
python  ${path}/main.py 4381 &
python  ${path}/main.py 4382 &
python  ${path}/main.py 4383 &
python  ${path}/main.py 4384 &
python  ${path}/main.py 4385 &
python  ${path}/main.py 4386 &
python  ${path}/main.py 4387 &
python  ${path}/main.py 4388 &
python  ${path}/main.py 4389 &
python  ${path}/main.py 4390 &
python  ${path}/main.py 4391 &
wait
