#!/bin/bash -l
#SBATCH -J RNN_55
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1944 &
python  ${path}/main.py 1945 &
python  ${path}/main.py 1946 &
python  ${path}/main.py 1947 &
python  ${path}/main.py 1948 &
python  ${path}/main.py 1949 &
python  ${path}/main.py 1950 &
python  ${path}/main.py 1951 &
python  ${path}/main.py 1952 &
python  ${path}/main.py 1953 &
python  ${path}/main.py 1954 &
python  ${path}/main.py 1955 &
python  ${path}/main.py 1956 &
python  ${path}/main.py 1957 &
python  ${path}/main.py 1958 &
python  ${path}/main.py 1959 &
python  ${path}/main.py 1960 &
python  ${path}/main.py 1961 &
python  ${path}/main.py 1962 &
python  ${path}/main.py 1963 &
python  ${path}/main.py 1964 &
python  ${path}/main.py 1965 &
python  ${path}/main.py 1966 &
python  ${path}/main.py 1967 &
python  ${path}/main.py 1968 &
python  ${path}/main.py 1969 &
python  ${path}/main.py 1970 &
python  ${path}/main.py 1971 &
python  ${path}/main.py 1972 &
python  ${path}/main.py 1973 &
python  ${path}/main.py 1974 &
python  ${path}/main.py 1975 &
python  ${path}/main.py 1976 &
python  ${path}/main.py 1977 &
python  ${path}/main.py 1978 &
python  ${path}/main.py 1979 &
wait
