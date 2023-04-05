#!/bin/bash -l
#SBATCH -J NN_34
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2376 &
python  ${path}/main.py 2377 &
python  ${path}/main.py 2378 &
python  ${path}/main.py 2379 &
python  ${path}/main.py 2380 &
python  ${path}/main.py 2381 &
python  ${path}/main.py 2382 &
python  ${path}/main.py 2383 &
python  ${path}/main.py 2384 &
python  ${path}/main.py 2385 &
python  ${path}/main.py 2386 &
python  ${path}/main.py 2387 &
python  ${path}/main.py 2388 &
python  ${path}/main.py 2389 &
python  ${path}/main.py 2390 &
python  ${path}/main.py 2391 &
python  ${path}/main.py 2392 &
python  ${path}/main.py 2393 &
python  ${path}/main.py 2394 &
python  ${path}/main.py 2395 &
python  ${path}/main.py 2396 &
python  ${path}/main.py 2397 &
python  ${path}/main.py 2398 &
python  ${path}/main.py 2399 &
python  ${path}/main.py 2400 &
python  ${path}/main.py 2401 &
python  ${path}/main.py 2402 &
python  ${path}/main.py 2403 &
python  ${path}/main.py 2404 &
python  ${path}/main.py 2405 &
python  ${path}/main.py 2406 &
python  ${path}/main.py 2407 &
python  ${path}/main.py 2408 &
python  ${path}/main.py 2409 &
python  ${path}/main.py 2410 &
python  ${path}/main.py 2411 &
python  ${path}/main.py 2412 &
python  ${path}/main.py 2413 &
python  ${path}/main.py 2414 &
python  ${path}/main.py 2415 &
python  ${path}/main.py 2416 &
python  ${path}/main.py 2417 &
python  ${path}/main.py 2418 &
python  ${path}/main.py 2419 &
python  ${path}/main.py 2420 &
python  ${path}/main.py 2421 &
python  ${path}/main.py 2422 &
python  ${path}/main.py 2423 &
python  ${path}/main.py 2424 &
python  ${path}/main.py 2425 &
python  ${path}/main.py 2426 &
python  ${path}/main.py 2427 &
python  ${path}/main.py 2428 &
python  ${path}/main.py 2429 &
python  ${path}/main.py 2430 &
python  ${path}/main.py 2431 &
python  ${path}/main.py 2432 &
python  ${path}/main.py 2433 &
python  ${path}/main.py 2434 &
python  ${path}/main.py 2435 &
python  ${path}/main.py 2436 &
python  ${path}/main.py 2437 &
python  ${path}/main.py 2438 &
python  ${path}/main.py 2439 &
python  ${path}/main.py 2440 &
python  ${path}/main.py 2441 &
python  ${path}/main.py 2442 &
python  ${path}/main.py 2443 &
python  ${path}/main.py 2444 &
python  ${path}/main.py 2445 &
python  ${path}/main.py 2446 &
python  ${path}/main.py 2447 &
wait
