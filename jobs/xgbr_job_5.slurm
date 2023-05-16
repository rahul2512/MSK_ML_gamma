#!/bin/bash -l
#SBATCH -J xgbr_5
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 288 &
python  ${path}/main.py 289 &
python  ${path}/main.py 290 &
python  ${path}/main.py 291 &
python  ${path}/main.py 292 &
python  ${path}/main.py 293 &
python  ${path}/main.py 294 &
python  ${path}/main.py 295 &
python  ${path}/main.py 296 &
python  ${path}/main.py 297 &
python  ${path}/main.py 298 &
python  ${path}/main.py 299 &
python  ${path}/main.py 300 &
python  ${path}/main.py 301 &
python  ${path}/main.py 302 &
python  ${path}/main.py 303 &
python  ${path}/main.py 304 &
python  ${path}/main.py 305 &
python  ${path}/main.py 306 &
python  ${path}/main.py 307 &
python  ${path}/main.py 308 &
python  ${path}/main.py 309 &
python  ${path}/main.py 310 &
python  ${path}/main.py 311 &
python  ${path}/main.py 312 &
python  ${path}/main.py 313 &
python  ${path}/main.py 314 &
python  ${path}/main.py 315 &
python  ${path}/main.py 316 &
python  ${path}/main.py 317 &
python  ${path}/main.py 318 &
python  ${path}/main.py 319 &
python  ${path}/main.py 320 &
python  ${path}/main.py 321 &
python  ${path}/main.py 322 &
python  ${path}/main.py 323 &
python  ${path}/main.py 324 &
python  ${path}/main.py 325 &
python  ${path}/main.py 326 &
python  ${path}/main.py 327 &
python  ${path}/main.py 328 &
python  ${path}/main.py 329 &
python  ${path}/main.py 330 &
python  ${path}/main.py 331 &
python  ${path}/main.py 332 &
python  ${path}/main.py 333 &
python  ${path}/main.py 334 &
python  ${path}/main.py 335 &
python  ${path}/main.py 336 &
python  ${path}/main.py 337 &
python  ${path}/main.py 338 &
python  ${path}/main.py 339 &
python  ${path}/main.py 340 &
python  ${path}/main.py 341 &
python  ${path}/main.py 342 &
python  ${path}/main.py 343 &
python  ${path}/main.py 344 &
python  ${path}/main.py 345 &
python  ${path}/main.py 346 &
python  ${path}/main.py 347 &
python  ${path}/main.py 348 &
python  ${path}/main.py 349 &
python  ${path}/main.py 350 &
python  ${path}/main.py 351 &
python  ${path}/main.py 352 &
python  ${path}/main.py 353 &
python  ${path}/main.py 354 &
python  ${path}/main.py 355 &
python  ${path}/main.py 356 &
python  ${path}/main.py 357 &
python  ${path}/main.py 358 &
python  ${path}/main.py 359 &
wait
