#!/bin/bash -l
#SBATCH -J NN_20
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1368 &
python  ${path}/main.py 1369 &
python  ${path}/main.py 1370 &
python  ${path}/main.py 1371 &
python  ${path}/main.py 1372 &
python  ${path}/main.py 1373 &
python  ${path}/main.py 1374 &
python  ${path}/main.py 1375 &
python  ${path}/main.py 1376 &
python  ${path}/main.py 1377 &
python  ${path}/main.py 1378 &
python  ${path}/main.py 1379 &
python  ${path}/main.py 1380 &
python  ${path}/main.py 1381 &
python  ${path}/main.py 1382 &
python  ${path}/main.py 1383 &
python  ${path}/main.py 1384 &
python  ${path}/main.py 1385 &
python  ${path}/main.py 1386 &
python  ${path}/main.py 1387 &
python  ${path}/main.py 1388 &
python  ${path}/main.py 1389 &
python  ${path}/main.py 1390 &
python  ${path}/main.py 1391 &
python  ${path}/main.py 1392 &
python  ${path}/main.py 1393 &
python  ${path}/main.py 1394 &
python  ${path}/main.py 1395 &
python  ${path}/main.py 1396 &
python  ${path}/main.py 1397 &
python  ${path}/main.py 1398 &
python  ${path}/main.py 1399 &
python  ${path}/main.py 1400 &
python  ${path}/main.py 1401 &
python  ${path}/main.py 1402 &
python  ${path}/main.py 1403 &
python  ${path}/main.py 1404 &
python  ${path}/main.py 1405 &
python  ${path}/main.py 1406 &
python  ${path}/main.py 1407 &
python  ${path}/main.py 1408 &
python  ${path}/main.py 1409 &
python  ${path}/main.py 1410 &
python  ${path}/main.py 1411 &
python  ${path}/main.py 1412 &
python  ${path}/main.py 1413 &
python  ${path}/main.py 1414 &
python  ${path}/main.py 1415 &
python  ${path}/main.py 1416 &
python  ${path}/main.py 1417 &
python  ${path}/main.py 1418 &
python  ${path}/main.py 1419 &
python  ${path}/main.py 1420 &
python  ${path}/main.py 1421 &
python  ${path}/main.py 1422 &
python  ${path}/main.py 1423 &
python  ${path}/main.py 1424 &
python  ${path}/main.py 1425 &
python  ${path}/main.py 1426 &
python  ${path}/main.py 1427 &
python  ${path}/main.py 1428 &
python  ${path}/main.py 1429 &
python  ${path}/main.py 1430 &
python  ${path}/main.py 1431 &
python  ${path}/main.py 1432 &
python  ${path}/main.py 1433 &
python  ${path}/main.py 1434 &
python  ${path}/main.py 1435 &
python  ${path}/main.py 1436 &
python  ${path}/main.py 1437 &
python  ${path}/main.py 1438 &
python  ${path}/main.py 1439 &
wait
