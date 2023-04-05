#!/bin/bash -l
#SBATCH -J NN_144
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10296 &
python  ${path}/main.py 10297 &
python  ${path}/main.py 10298 &
python  ${path}/main.py 10299 &
python  ${path}/main.py 10300 &
python  ${path}/main.py 10301 &
python  ${path}/main.py 10302 &
python  ${path}/main.py 10303 &
python  ${path}/main.py 10304 &
python  ${path}/main.py 10305 &
python  ${path}/main.py 10306 &
python  ${path}/main.py 10307 &
python  ${path}/main.py 10308 &
python  ${path}/main.py 10309 &
python  ${path}/main.py 10310 &
python  ${path}/main.py 10311 &
python  ${path}/main.py 10312 &
python  ${path}/main.py 10313 &
python  ${path}/main.py 10314 &
python  ${path}/main.py 10315 &
python  ${path}/main.py 10316 &
python  ${path}/main.py 10317 &
python  ${path}/main.py 10318 &
python  ${path}/main.py 10319 &
python  ${path}/main.py 10320 &
python  ${path}/main.py 10321 &
python  ${path}/main.py 10322 &
python  ${path}/main.py 10323 &
python  ${path}/main.py 10324 &
python  ${path}/main.py 10325 &
python  ${path}/main.py 10326 &
python  ${path}/main.py 10327 &
python  ${path}/main.py 10328 &
python  ${path}/main.py 10329 &
python  ${path}/main.py 10330 &
python  ${path}/main.py 10331 &
python  ${path}/main.py 10332 &
python  ${path}/main.py 10333 &
python  ${path}/main.py 10334 &
python  ${path}/main.py 10335 &
python  ${path}/main.py 10336 &
python  ${path}/main.py 10337 &
python  ${path}/main.py 10338 &
python  ${path}/main.py 10339 &
python  ${path}/main.py 10340 &
python  ${path}/main.py 10341 &
python  ${path}/main.py 10342 &
python  ${path}/main.py 10343 &
python  ${path}/main.py 10344 &
python  ${path}/main.py 10345 &
python  ${path}/main.py 10346 &
python  ${path}/main.py 10347 &
python  ${path}/main.py 10348 &
python  ${path}/main.py 10349 &
python  ${path}/main.py 10350 &
python  ${path}/main.py 10351 &
python  ${path}/main.py 10352 &
python  ${path}/main.py 10353 &
python  ${path}/main.py 10354 &
python  ${path}/main.py 10355 &
python  ${path}/main.py 10356 &
python  ${path}/main.py 10357 &
python  ${path}/main.py 10358 &
python  ${path}/main.py 10359 &
python  ${path}/main.py 10360 &
python  ${path}/main.py 10361 &
python  ${path}/main.py 10362 &
python  ${path}/main.py 10363 &
python  ${path}/main.py 10364 &
python  ${path}/main.py 10365 &
python  ${path}/main.py 10366 &
python  ${path}/main.py 10367 &
wait
