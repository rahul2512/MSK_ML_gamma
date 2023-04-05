#!/bin/bash -l
#SBATCH -J NN_76
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5400 &
python  ${path}/main.py 5401 &
python  ${path}/main.py 5402 &
python  ${path}/main.py 5403 &
python  ${path}/main.py 5404 &
python  ${path}/main.py 5405 &
python  ${path}/main.py 5406 &
python  ${path}/main.py 5407 &
python  ${path}/main.py 5408 &
python  ${path}/main.py 5409 &
python  ${path}/main.py 5410 &
python  ${path}/main.py 5411 &
python  ${path}/main.py 5412 &
python  ${path}/main.py 5413 &
python  ${path}/main.py 5414 &
python  ${path}/main.py 5415 &
python  ${path}/main.py 5416 &
python  ${path}/main.py 5417 &
python  ${path}/main.py 5418 &
python  ${path}/main.py 5419 &
python  ${path}/main.py 5420 &
python  ${path}/main.py 5421 &
python  ${path}/main.py 5422 &
python  ${path}/main.py 5423 &
python  ${path}/main.py 5424 &
python  ${path}/main.py 5425 &
python  ${path}/main.py 5426 &
python  ${path}/main.py 5427 &
python  ${path}/main.py 5428 &
python  ${path}/main.py 5429 &
python  ${path}/main.py 5430 &
python  ${path}/main.py 5431 &
python  ${path}/main.py 5432 &
python  ${path}/main.py 5433 &
python  ${path}/main.py 5434 &
python  ${path}/main.py 5435 &
python  ${path}/main.py 5436 &
python  ${path}/main.py 5437 &
python  ${path}/main.py 5438 &
python  ${path}/main.py 5439 &
python  ${path}/main.py 5440 &
python  ${path}/main.py 5441 &
python  ${path}/main.py 5442 &
python  ${path}/main.py 5443 &
python  ${path}/main.py 5444 &
python  ${path}/main.py 5445 &
python  ${path}/main.py 5446 &
python  ${path}/main.py 5447 &
python  ${path}/main.py 5448 &
python  ${path}/main.py 5449 &
python  ${path}/main.py 5450 &
python  ${path}/main.py 5451 &
python  ${path}/main.py 5452 &
python  ${path}/main.py 5453 &
python  ${path}/main.py 5454 &
python  ${path}/main.py 5455 &
python  ${path}/main.py 5456 &
python  ${path}/main.py 5457 &
python  ${path}/main.py 5458 &
python  ${path}/main.py 5459 &
python  ${path}/main.py 5460 &
python  ${path}/main.py 5461 &
python  ${path}/main.py 5462 &
python  ${path}/main.py 5463 &
python  ${path}/main.py 5464 &
python  ${path}/main.py 5465 &
python  ${path}/main.py 5466 &
python  ${path}/main.py 5467 &
python  ${path}/main.py 5468 &
python  ${path}/main.py 5469 &
python  ${path}/main.py 5470 &
python  ${path}/main.py 5471 &
wait
