#!/bin/bash -l
#SBATCH -J CNNLSTM_23
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1584 &
python  ${path}/main.py 1585 &
python  ${path}/main.py 1586 &
python  ${path}/main.py 1587 &
python  ${path}/main.py 1588 &
python  ${path}/main.py 1589 &
python  ${path}/main.py 1590 &
python  ${path}/main.py 1591 &
python  ${path}/main.py 1592 &
python  ${path}/main.py 1593 &
python  ${path}/main.py 1594 &
python  ${path}/main.py 1595 &
python  ${path}/main.py 1596 &
python  ${path}/main.py 1597 &
python  ${path}/main.py 1598 &
python  ${path}/main.py 1599 &
python  ${path}/main.py 1600 &
python  ${path}/main.py 1601 &
python  ${path}/main.py 1602 &
python  ${path}/main.py 1603 &
python  ${path}/main.py 1604 &
python  ${path}/main.py 1605 &
python  ${path}/main.py 1606 &
python  ${path}/main.py 1607 &
python  ${path}/main.py 1608 &
python  ${path}/main.py 1609 &
python  ${path}/main.py 1610 &
python  ${path}/main.py 1611 &
python  ${path}/main.py 1612 &
python  ${path}/main.py 1613 &
python  ${path}/main.py 1614 &
python  ${path}/main.py 1615 &
python  ${path}/main.py 1616 &
python  ${path}/main.py 1617 &
python  ${path}/main.py 1618 &
python  ${path}/main.py 1619 &
python  ${path}/main.py 1620 &
python  ${path}/main.py 1621 &
python  ${path}/main.py 1622 &
python  ${path}/main.py 1623 &
python  ${path}/main.py 1624 &
python  ${path}/main.py 1625 &
python  ${path}/main.py 1626 &
python  ${path}/main.py 1627 &
python  ${path}/main.py 1628 &
python  ${path}/main.py 1629 &
python  ${path}/main.py 1630 &
python  ${path}/main.py 1631 &
python  ${path}/main.py 1632 &
python  ${path}/main.py 1633 &
python  ${path}/main.py 1634 &
python  ${path}/main.py 1635 &
python  ${path}/main.py 1636 &
python  ${path}/main.py 1637 &
python  ${path}/main.py 1638 &
python  ${path}/main.py 1639 &
python  ${path}/main.py 1640 &
python  ${path}/main.py 1641 &
python  ${path}/main.py 1642 &
python  ${path}/main.py 1643 &
python  ${path}/main.py 1644 &
python  ${path}/main.py 1645 &
python  ${path}/main.py 1646 &
python  ${path}/main.py 1647 &
python  ${path}/main.py 1648 &
python  ${path}/main.py 1649 &
python  ${path}/main.py 1650 &
python  ${path}/main.py 1651 &
python  ${path}/main.py 1652 &
python  ${path}/main.py 1653 &
python  ${path}/main.py 1654 &
python  ${path}/main.py 1655 &
wait
