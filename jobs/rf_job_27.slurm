#!/bin/bash -l
#SBATCH -J rf_27
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1872 &
python  ${path}/main.py 1873 &
python  ${path}/main.py 1874 &
python  ${path}/main.py 1875 &
python  ${path}/main.py 1876 &
python  ${path}/main.py 1877 &
python  ${path}/main.py 1878 &
python  ${path}/main.py 1879 &
python  ${path}/main.py 1880 &
python  ${path}/main.py 1881 &
python  ${path}/main.py 1882 &
python  ${path}/main.py 1883 &
python  ${path}/main.py 1884 &
python  ${path}/main.py 1885 &
python  ${path}/main.py 1886 &
python  ${path}/main.py 1887 &
python  ${path}/main.py 1888 &
python  ${path}/main.py 1889 &
python  ${path}/main.py 1890 &
python  ${path}/main.py 1891 &
python  ${path}/main.py 1892 &
python  ${path}/main.py 1893 &
python  ${path}/main.py 1894 &
python  ${path}/main.py 1895 &
python  ${path}/main.py 1896 &
python  ${path}/main.py 1897 &
python  ${path}/main.py 1898 &
python  ${path}/main.py 1899 &
python  ${path}/main.py 1900 &
python  ${path}/main.py 1901 &
python  ${path}/main.py 1902 &
python  ${path}/main.py 1903 &
python  ${path}/main.py 1904 &
python  ${path}/main.py 1905 &
python  ${path}/main.py 1906 &
python  ${path}/main.py 1907 &
python  ${path}/main.py 1908 &
python  ${path}/main.py 1909 &
python  ${path}/main.py 1910 &
python  ${path}/main.py 1911 &
python  ${path}/main.py 1912 &
python  ${path}/main.py 1913 &
python  ${path}/main.py 1914 &
python  ${path}/main.py 1915 &
python  ${path}/main.py 1916 &
python  ${path}/main.py 1917 &
python  ${path}/main.py 1918 &
python  ${path}/main.py 1919 &
python  ${path}/main.py 1920 &
python  ${path}/main.py 1921 &
python  ${path}/main.py 1922 &
python  ${path}/main.py 1923 &
python  ${path}/main.py 1924 &
python  ${path}/main.py 1925 &
python  ${path}/main.py 1926 &
python  ${path}/main.py 1927 &
python  ${path}/main.py 1928 &
python  ${path}/main.py 1929 &
python  ${path}/main.py 1930 &
python  ${path}/main.py 1931 &
python  ${path}/main.py 1932 &
python  ${path}/main.py 1933 &
python  ${path}/main.py 1934 &
python  ${path}/main.py 1935 &
python  ${path}/main.py 1936 &
python  ${path}/main.py 1937 &
python  ${path}/main.py 1938 &
python  ${path}/main.py 1939 &
python  ${path}/main.py 1940 &
python  ${path}/main.py 1941 &
python  ${path}/main.py 1942 &
python  ${path}/main.py 1943 &
wait
