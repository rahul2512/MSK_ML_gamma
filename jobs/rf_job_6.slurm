#!/bin/bash -l
#SBATCH -J rf_6
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 360 &
python  ${path}/main.py 361 &
python  ${path}/main.py 362 &
python  ${path}/main.py 363 &
python  ${path}/main.py 364 &
python  ${path}/main.py 365 &
python  ${path}/main.py 366 &
python  ${path}/main.py 367 &
python  ${path}/main.py 368 &
python  ${path}/main.py 369 &
python  ${path}/main.py 370 &
python  ${path}/main.py 371 &
python  ${path}/main.py 372 &
python  ${path}/main.py 373 &
python  ${path}/main.py 374 &
python  ${path}/main.py 375 &
python  ${path}/main.py 376 &
python  ${path}/main.py 377 &
python  ${path}/main.py 378 &
python  ${path}/main.py 379 &
python  ${path}/main.py 380 &
python  ${path}/main.py 381 &
python  ${path}/main.py 382 &
python  ${path}/main.py 383 &
python  ${path}/main.py 384 &
python  ${path}/main.py 385 &
python  ${path}/main.py 386 &
python  ${path}/main.py 387 &
python  ${path}/main.py 388 &
python  ${path}/main.py 389 &
python  ${path}/main.py 390 &
python  ${path}/main.py 391 &
python  ${path}/main.py 392 &
python  ${path}/main.py 393 &
python  ${path}/main.py 394 &
python  ${path}/main.py 395 &
python  ${path}/main.py 396 &
python  ${path}/main.py 397 &
python  ${path}/main.py 398 &
python  ${path}/main.py 399 &
python  ${path}/main.py 400 &
python  ${path}/main.py 401 &
python  ${path}/main.py 402 &
python  ${path}/main.py 403 &
python  ${path}/main.py 404 &
python  ${path}/main.py 405 &
python  ${path}/main.py 406 &
python  ${path}/main.py 407 &
python  ${path}/main.py 408 &
python  ${path}/main.py 409 &
python  ${path}/main.py 410 &
python  ${path}/main.py 411 &
python  ${path}/main.py 412 &
python  ${path}/main.py 413 &
python  ${path}/main.py 414 &
python  ${path}/main.py 415 &
python  ${path}/main.py 416 &
python  ${path}/main.py 417 &
python  ${path}/main.py 418 &
python  ${path}/main.py 419 &
python  ${path}/main.py 420 &
python  ${path}/main.py 421 &
python  ${path}/main.py 422 &
python  ${path}/main.py 423 &
python  ${path}/main.py 424 &
python  ${path}/main.py 425 &
python  ${path}/main.py 426 &
python  ${path}/main.py 427 &
python  ${path}/main.py 428 &
python  ${path}/main.py 429 &
python  ${path}/main.py 430 &
python  ${path}/main.py 431 &
wait
