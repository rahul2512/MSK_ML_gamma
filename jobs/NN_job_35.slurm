#!/bin/bash -l
#SBATCH -J NN_35
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2448 &
python  ${path}/main.py 2449 &
python  ${path}/main.py 2450 &
python  ${path}/main.py 2451 &
python  ${path}/main.py 2452 &
python  ${path}/main.py 2453 &
python  ${path}/main.py 2454 &
python  ${path}/main.py 2455 &
python  ${path}/main.py 2456 &
python  ${path}/main.py 2457 &
python  ${path}/main.py 2458 &
python  ${path}/main.py 2459 &
python  ${path}/main.py 2460 &
python  ${path}/main.py 2461 &
python  ${path}/main.py 2462 &
python  ${path}/main.py 2463 &
python  ${path}/main.py 2464 &
python  ${path}/main.py 2465 &
python  ${path}/main.py 2466 &
python  ${path}/main.py 2467 &
python  ${path}/main.py 2468 &
python  ${path}/main.py 2469 &
python  ${path}/main.py 2470 &
python  ${path}/main.py 2471 &
python  ${path}/main.py 2472 &
python  ${path}/main.py 2473 &
python  ${path}/main.py 2474 &
python  ${path}/main.py 2475 &
python  ${path}/main.py 2476 &
python  ${path}/main.py 2477 &
python  ${path}/main.py 2478 &
python  ${path}/main.py 2479 &
python  ${path}/main.py 2480 &
python  ${path}/main.py 2481 &
python  ${path}/main.py 2482 &
python  ${path}/main.py 2483 &
python  ${path}/main.py 2484 &
python  ${path}/main.py 2485 &
python  ${path}/main.py 2486 &
python  ${path}/main.py 2487 &
python  ${path}/main.py 2488 &
python  ${path}/main.py 2489 &
python  ${path}/main.py 2490 &
python  ${path}/main.py 2491 &
python  ${path}/main.py 2492 &
python  ${path}/main.py 2493 &
python  ${path}/main.py 2494 &
python  ${path}/main.py 2495 &
python  ${path}/main.py 2496 &
python  ${path}/main.py 2497 &
python  ${path}/main.py 2498 &
python  ${path}/main.py 2499 &
python  ${path}/main.py 2500 &
python  ${path}/main.py 2501 &
python  ${path}/main.py 2502 &
python  ${path}/main.py 2503 &
python  ${path}/main.py 2504 &
python  ${path}/main.py 2505 &
python  ${path}/main.py 2506 &
python  ${path}/main.py 2507 &
python  ${path}/main.py 2508 &
python  ${path}/main.py 2509 &
python  ${path}/main.py 2510 &
python  ${path}/main.py 2511 &
python  ${path}/main.py 2512 &
python  ${path}/main.py 2513 &
python  ${path}/main.py 2514 &
python  ${path}/main.py 2515 &
python  ${path}/main.py 2516 &
python  ${path}/main.py 2517 &
python  ${path}/main.py 2518 &
python  ${path}/main.py 2519 &
wait
