#!/bin/bash -l
#SBATCH -J NN_25
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1728 &
python  ${path}/main.py 1729 &
python  ${path}/main.py 1730 &
python  ${path}/main.py 1731 &
python  ${path}/main.py 1732 &
python  ${path}/main.py 1733 &
python  ${path}/main.py 1734 &
python  ${path}/main.py 1735 &
python  ${path}/main.py 1736 &
python  ${path}/main.py 1737 &
python  ${path}/main.py 1738 &
python  ${path}/main.py 1739 &
python  ${path}/main.py 1740 &
python  ${path}/main.py 1741 &
python  ${path}/main.py 1742 &
python  ${path}/main.py 1743 &
python  ${path}/main.py 1744 &
python  ${path}/main.py 1745 &
python  ${path}/main.py 1746 &
python  ${path}/main.py 1747 &
python  ${path}/main.py 1748 &
python  ${path}/main.py 1749 &
python  ${path}/main.py 1750 &
python  ${path}/main.py 1751 &
python  ${path}/main.py 1752 &
python  ${path}/main.py 1753 &
python  ${path}/main.py 1754 &
python  ${path}/main.py 1755 &
python  ${path}/main.py 1756 &
python  ${path}/main.py 1757 &
python  ${path}/main.py 1758 &
python  ${path}/main.py 1759 &
python  ${path}/main.py 1760 &
python  ${path}/main.py 1761 &
python  ${path}/main.py 1762 &
python  ${path}/main.py 1763 &
python  ${path}/main.py 1764 &
python  ${path}/main.py 1765 &
python  ${path}/main.py 1766 &
python  ${path}/main.py 1767 &
python  ${path}/main.py 1768 &
python  ${path}/main.py 1769 &
python  ${path}/main.py 1770 &
python  ${path}/main.py 1771 &
python  ${path}/main.py 1772 &
python  ${path}/main.py 1773 &
python  ${path}/main.py 1774 &
python  ${path}/main.py 1775 &
python  ${path}/main.py 1776 &
python  ${path}/main.py 1777 &
python  ${path}/main.py 1778 &
python  ${path}/main.py 1779 &
python  ${path}/main.py 1780 &
python  ${path}/main.py 1781 &
python  ${path}/main.py 1782 &
python  ${path}/main.py 1783 &
python  ${path}/main.py 1784 &
python  ${path}/main.py 1785 &
python  ${path}/main.py 1786 &
python  ${path}/main.py 1787 &
python  ${path}/main.py 1788 &
python  ${path}/main.py 1789 &
python  ${path}/main.py 1790 &
python  ${path}/main.py 1791 &
python  ${path}/main.py 1792 &
python  ${path}/main.py 1793 &
python  ${path}/main.py 1794 &
python  ${path}/main.py 1795 &
python  ${path}/main.py 1796 &
python  ${path}/main.py 1797 &
python  ${path}/main.py 1798 &
python  ${path}/main.py 1799 &
wait
