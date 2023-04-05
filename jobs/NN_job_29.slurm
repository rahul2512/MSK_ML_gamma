#!/bin/bash -l
#SBATCH -J NN_29
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2016 &
python  ${path}/main.py 2017 &
python  ${path}/main.py 2018 &
python  ${path}/main.py 2019 &
python  ${path}/main.py 2020 &
python  ${path}/main.py 2021 &
python  ${path}/main.py 2022 &
python  ${path}/main.py 2023 &
python  ${path}/main.py 2024 &
python  ${path}/main.py 2025 &
python  ${path}/main.py 2026 &
python  ${path}/main.py 2027 &
python  ${path}/main.py 2028 &
python  ${path}/main.py 2029 &
python  ${path}/main.py 2030 &
python  ${path}/main.py 2031 &
python  ${path}/main.py 2032 &
python  ${path}/main.py 2033 &
python  ${path}/main.py 2034 &
python  ${path}/main.py 2035 &
python  ${path}/main.py 2036 &
python  ${path}/main.py 2037 &
python  ${path}/main.py 2038 &
python  ${path}/main.py 2039 &
python  ${path}/main.py 2040 &
python  ${path}/main.py 2041 &
python  ${path}/main.py 2042 &
python  ${path}/main.py 2043 &
python  ${path}/main.py 2044 &
python  ${path}/main.py 2045 &
python  ${path}/main.py 2046 &
python  ${path}/main.py 2047 &
python  ${path}/main.py 2048 &
python  ${path}/main.py 2049 &
python  ${path}/main.py 2050 &
python  ${path}/main.py 2051 &
python  ${path}/main.py 2052 &
python  ${path}/main.py 2053 &
python  ${path}/main.py 2054 &
python  ${path}/main.py 2055 &
python  ${path}/main.py 2056 &
python  ${path}/main.py 2057 &
python  ${path}/main.py 2058 &
python  ${path}/main.py 2059 &
python  ${path}/main.py 2060 &
python  ${path}/main.py 2061 &
python  ${path}/main.py 2062 &
python  ${path}/main.py 2063 &
python  ${path}/main.py 2064 &
python  ${path}/main.py 2065 &
python  ${path}/main.py 2066 &
python  ${path}/main.py 2067 &
python  ${path}/main.py 2068 &
python  ${path}/main.py 2069 &
python  ${path}/main.py 2070 &
python  ${path}/main.py 2071 &
python  ${path}/main.py 2072 &
python  ${path}/main.py 2073 &
python  ${path}/main.py 2074 &
python  ${path}/main.py 2075 &
python  ${path}/main.py 2076 &
python  ${path}/main.py 2077 &
python  ${path}/main.py 2078 &
python  ${path}/main.py 2079 &
python  ${path}/main.py 2080 &
python  ${path}/main.py 2081 &
python  ${path}/main.py 2082 &
python  ${path}/main.py 2083 &
python  ${path}/main.py 2084 &
python  ${path}/main.py 2085 &
python  ${path}/main.py 2086 &
python  ${path}/main.py 2087 &
wait
