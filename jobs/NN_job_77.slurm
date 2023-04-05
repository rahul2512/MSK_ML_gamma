#!/bin/bash -l
#SBATCH -J NN_77
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5472 &
python  ${path}/main.py 5473 &
python  ${path}/main.py 5474 &
python  ${path}/main.py 5475 &
python  ${path}/main.py 5476 &
python  ${path}/main.py 5477 &
python  ${path}/main.py 5478 &
python  ${path}/main.py 5479 &
python  ${path}/main.py 5480 &
python  ${path}/main.py 5481 &
python  ${path}/main.py 5482 &
python  ${path}/main.py 5483 &
python  ${path}/main.py 5484 &
python  ${path}/main.py 5485 &
python  ${path}/main.py 5486 &
python  ${path}/main.py 5487 &
python  ${path}/main.py 5488 &
python  ${path}/main.py 5489 &
python  ${path}/main.py 5490 &
python  ${path}/main.py 5491 &
python  ${path}/main.py 5492 &
python  ${path}/main.py 5493 &
python  ${path}/main.py 5494 &
python  ${path}/main.py 5495 &
python  ${path}/main.py 5496 &
python  ${path}/main.py 5497 &
python  ${path}/main.py 5498 &
python  ${path}/main.py 5499 &
python  ${path}/main.py 5500 &
python  ${path}/main.py 5501 &
python  ${path}/main.py 5502 &
python  ${path}/main.py 5503 &
python  ${path}/main.py 5504 &
python  ${path}/main.py 5505 &
python  ${path}/main.py 5506 &
python  ${path}/main.py 5507 &
python  ${path}/main.py 5508 &
python  ${path}/main.py 5509 &
python  ${path}/main.py 5510 &
python  ${path}/main.py 5511 &
python  ${path}/main.py 5512 &
python  ${path}/main.py 5513 &
python  ${path}/main.py 5514 &
python  ${path}/main.py 5515 &
python  ${path}/main.py 5516 &
python  ${path}/main.py 5517 &
python  ${path}/main.py 5518 &
python  ${path}/main.py 5519 &
python  ${path}/main.py 5520 &
python  ${path}/main.py 5521 &
python  ${path}/main.py 5522 &
python  ${path}/main.py 5523 &
python  ${path}/main.py 5524 &
python  ${path}/main.py 5525 &
python  ${path}/main.py 5526 &
python  ${path}/main.py 5527 &
python  ${path}/main.py 5528 &
python  ${path}/main.py 5529 &
python  ${path}/main.py 5530 &
python  ${path}/main.py 5531 &
python  ${path}/main.py 5532 &
python  ${path}/main.py 5533 &
python  ${path}/main.py 5534 &
python  ${path}/main.py 5535 &
python  ${path}/main.py 5536 &
python  ${path}/main.py 5537 &
python  ${path}/main.py 5538 &
python  ${path}/main.py 5539 &
python  ${path}/main.py 5540 &
python  ${path}/main.py 5541 &
python  ${path}/main.py 5542 &
python  ${path}/main.py 5543 &
wait
