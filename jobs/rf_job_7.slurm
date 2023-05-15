#!/bin/bash -l
#SBATCH -J rf_7
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 432 &
python  ${path}/main.py 433 &
python  ${path}/main.py 434 &
python  ${path}/main.py 435 &
python  ${path}/main.py 436 &
python  ${path}/main.py 437 &
python  ${path}/main.py 438 &
python  ${path}/main.py 439 &
python  ${path}/main.py 440 &
python  ${path}/main.py 441 &
python  ${path}/main.py 442 &
python  ${path}/main.py 443 &
python  ${path}/main.py 444 &
python  ${path}/main.py 445 &
python  ${path}/main.py 446 &
python  ${path}/main.py 447 &
python  ${path}/main.py 448 &
python  ${path}/main.py 449 &
python  ${path}/main.py 450 &
python  ${path}/main.py 451 &
python  ${path}/main.py 452 &
python  ${path}/main.py 453 &
python  ${path}/main.py 454 &
python  ${path}/main.py 455 &
python  ${path}/main.py 456 &
python  ${path}/main.py 457 &
python  ${path}/main.py 458 &
python  ${path}/main.py 459 &
python  ${path}/main.py 460 &
python  ${path}/main.py 461 &
python  ${path}/main.py 462 &
python  ${path}/main.py 463 &
python  ${path}/main.py 464 &
python  ${path}/main.py 465 &
python  ${path}/main.py 466 &
python  ${path}/main.py 467 &
python  ${path}/main.py 468 &
python  ${path}/main.py 469 &
python  ${path}/main.py 470 &
python  ${path}/main.py 471 &
python  ${path}/main.py 472 &
python  ${path}/main.py 473 &
python  ${path}/main.py 474 &
python  ${path}/main.py 475 &
python  ${path}/main.py 476 &
python  ${path}/main.py 477 &
python  ${path}/main.py 478 &
python  ${path}/main.py 479 &
python  ${path}/main.py 480 &
python  ${path}/main.py 481 &
python  ${path}/main.py 482 &
python  ${path}/main.py 483 &
python  ${path}/main.py 484 &
python  ${path}/main.py 485 &
python  ${path}/main.py 486 &
python  ${path}/main.py 487 &
python  ${path}/main.py 488 &
python  ${path}/main.py 489 &
python  ${path}/main.py 490 &
python  ${path}/main.py 491 &
python  ${path}/main.py 492 &
python  ${path}/main.py 493 &
python  ${path}/main.py 494 &
python  ${path}/main.py 495 &
python  ${path}/main.py 496 &
python  ${path}/main.py 497 &
python  ${path}/main.py 498 &
python  ${path}/main.py 499 &
python  ${path}/main.py 500 &
python  ${path}/main.py 501 &
python  ${path}/main.py 502 &
python  ${path}/main.py 503 &
wait
