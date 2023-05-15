#!/bin/bash -l
#SBATCH -J rf_21
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1440 &
python  ${path}/main.py 1441 &
python  ${path}/main.py 1442 &
python  ${path}/main.py 1443 &
python  ${path}/main.py 1444 &
python  ${path}/main.py 1445 &
python  ${path}/main.py 1446 &
python  ${path}/main.py 1447 &
python  ${path}/main.py 1448 &
python  ${path}/main.py 1449 &
python  ${path}/main.py 1450 &
python  ${path}/main.py 1451 &
python  ${path}/main.py 1452 &
python  ${path}/main.py 1453 &
python  ${path}/main.py 1454 &
python  ${path}/main.py 1455 &
python  ${path}/main.py 1456 &
python  ${path}/main.py 1457 &
python  ${path}/main.py 1458 &
python  ${path}/main.py 1459 &
python  ${path}/main.py 1460 &
python  ${path}/main.py 1461 &
python  ${path}/main.py 1462 &
python  ${path}/main.py 1463 &
python  ${path}/main.py 1464 &
python  ${path}/main.py 1465 &
python  ${path}/main.py 1466 &
python  ${path}/main.py 1467 &
python  ${path}/main.py 1468 &
python  ${path}/main.py 1469 &
python  ${path}/main.py 1470 &
python  ${path}/main.py 1471 &
python  ${path}/main.py 1472 &
python  ${path}/main.py 1473 &
python  ${path}/main.py 1474 &
python  ${path}/main.py 1475 &
python  ${path}/main.py 1476 &
python  ${path}/main.py 1477 &
python  ${path}/main.py 1478 &
python  ${path}/main.py 1479 &
python  ${path}/main.py 1480 &
python  ${path}/main.py 1481 &
python  ${path}/main.py 1482 &
python  ${path}/main.py 1483 &
python  ${path}/main.py 1484 &
python  ${path}/main.py 1485 &
python  ${path}/main.py 1486 &
python  ${path}/main.py 1487 &
python  ${path}/main.py 1488 &
python  ${path}/main.py 1489 &
python  ${path}/main.py 1490 &
python  ${path}/main.py 1491 &
python  ${path}/main.py 1492 &
python  ${path}/main.py 1493 &
python  ${path}/main.py 1494 &
python  ${path}/main.py 1495 &
python  ${path}/main.py 1496 &
python  ${path}/main.py 1497 &
python  ${path}/main.py 1498 &
python  ${path}/main.py 1499 &
python  ${path}/main.py 1500 &
python  ${path}/main.py 1501 &
python  ${path}/main.py 1502 &
python  ${path}/main.py 1503 &
python  ${path}/main.py 1504 &
python  ${path}/main.py 1505 &
python  ${path}/main.py 1506 &
python  ${path}/main.py 1507 &
python  ${path}/main.py 1508 &
python  ${path}/main.py 1509 &
python  ${path}/main.py 1510 &
python  ${path}/main.py 1511 &
wait
