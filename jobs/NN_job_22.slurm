#!/bin/bash -l
#SBATCH -J NN_22
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1512 &
python  ${path}/main.py 1513 &
python  ${path}/main.py 1514 &
python  ${path}/main.py 1515 &
python  ${path}/main.py 1516 &
python  ${path}/main.py 1517 &
python  ${path}/main.py 1518 &
python  ${path}/main.py 1519 &
python  ${path}/main.py 1520 &
python  ${path}/main.py 1521 &
python  ${path}/main.py 1522 &
python  ${path}/main.py 1523 &
python  ${path}/main.py 1524 &
python  ${path}/main.py 1525 &
python  ${path}/main.py 1526 &
python  ${path}/main.py 1527 &
python  ${path}/main.py 1528 &
python  ${path}/main.py 1529 &
python  ${path}/main.py 1530 &
python  ${path}/main.py 1531 &
python  ${path}/main.py 1532 &
python  ${path}/main.py 1533 &
python  ${path}/main.py 1534 &
python  ${path}/main.py 1535 &
python  ${path}/main.py 1536 &
python  ${path}/main.py 1537 &
python  ${path}/main.py 1538 &
python  ${path}/main.py 1539 &
python  ${path}/main.py 1540 &
python  ${path}/main.py 1541 &
python  ${path}/main.py 1542 &
python  ${path}/main.py 1543 &
python  ${path}/main.py 1544 &
python  ${path}/main.py 1545 &
python  ${path}/main.py 1546 &
python  ${path}/main.py 1547 &
python  ${path}/main.py 1548 &
python  ${path}/main.py 1549 &
python  ${path}/main.py 1550 &
python  ${path}/main.py 1551 &
python  ${path}/main.py 1552 &
python  ${path}/main.py 1553 &
python  ${path}/main.py 1554 &
python  ${path}/main.py 1555 &
python  ${path}/main.py 1556 &
python  ${path}/main.py 1557 &
python  ${path}/main.py 1558 &
python  ${path}/main.py 1559 &
python  ${path}/main.py 1560 &
python  ${path}/main.py 1561 &
python  ${path}/main.py 1562 &
python  ${path}/main.py 1563 &
python  ${path}/main.py 1564 &
python  ${path}/main.py 1565 &
python  ${path}/main.py 1566 &
python  ${path}/main.py 1567 &
python  ${path}/main.py 1568 &
python  ${path}/main.py 1569 &
python  ${path}/main.py 1570 &
python  ${path}/main.py 1571 &
python  ${path}/main.py 1572 &
python  ${path}/main.py 1573 &
python  ${path}/main.py 1574 &
python  ${path}/main.py 1575 &
python  ${path}/main.py 1576 &
python  ${path}/main.py 1577 &
python  ${path}/main.py 1578 &
python  ${path}/main.py 1579 &
python  ${path}/main.py 1580 &
python  ${path}/main.py 1581 &
python  ${path}/main.py 1582 &
python  ${path}/main.py 1583 &
wait
