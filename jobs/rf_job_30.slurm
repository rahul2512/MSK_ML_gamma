#!/bin/bash -l
#SBATCH -J rf_30
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2088 &
python  ${path}/main.py 2089 &
python  ${path}/main.py 2090 &
python  ${path}/main.py 2091 &
python  ${path}/main.py 2092 &
python  ${path}/main.py 2093 &
python  ${path}/main.py 2094 &
python  ${path}/main.py 2095 &
python  ${path}/main.py 2096 &
python  ${path}/main.py 2097 &
python  ${path}/main.py 2098 &
python  ${path}/main.py 2099 &
python  ${path}/main.py 2100 &
python  ${path}/main.py 2101 &
python  ${path}/main.py 2102 &
python  ${path}/main.py 2103 &
python  ${path}/main.py 2104 &
python  ${path}/main.py 2105 &
python  ${path}/main.py 2106 &
python  ${path}/main.py 2107 &
python  ${path}/main.py 2108 &
python  ${path}/main.py 2109 &
python  ${path}/main.py 2110 &
python  ${path}/main.py 2111 &
python  ${path}/main.py 2112 &
python  ${path}/main.py 2113 &
python  ${path}/main.py 2114 &
python  ${path}/main.py 2115 &
python  ${path}/main.py 2116 &
python  ${path}/main.py 2117 &
python  ${path}/main.py 2118 &
python  ${path}/main.py 2119 &
python  ${path}/main.py 2120 &
python  ${path}/main.py 2121 &
python  ${path}/main.py 2122 &
python  ${path}/main.py 2123 &
python  ${path}/main.py 2124 &
python  ${path}/main.py 2125 &
python  ${path}/main.py 2126 &
python  ${path}/main.py 2127 &
python  ${path}/main.py 2128 &
python  ${path}/main.py 2129 &
python  ${path}/main.py 2130 &
python  ${path}/main.py 2131 &
python  ${path}/main.py 2132 &
python  ${path}/main.py 2133 &
python  ${path}/main.py 2134 &
python  ${path}/main.py 2135 &
python  ${path}/main.py 2136 &
python  ${path}/main.py 2137 &
python  ${path}/main.py 2138 &
python  ${path}/main.py 2139 &
python  ${path}/main.py 2140 &
python  ${path}/main.py 2141 &
python  ${path}/main.py 2142 &
python  ${path}/main.py 2143 &
python  ${path}/main.py 2144 &
python  ${path}/main.py 2145 &
python  ${path}/main.py 2146 &
python  ${path}/main.py 2147 &
python  ${path}/main.py 2148 &
python  ${path}/main.py 2149 &
python  ${path}/main.py 2150 &
python  ${path}/main.py 2151 &
python  ${path}/main.py 2152 &
python  ${path}/main.py 2153 &
python  ${path}/main.py 2154 &
python  ${path}/main.py 2155 &
python  ${path}/main.py 2156 &
python  ${path}/main.py 2157 &
python  ${path}/main.py 2158 &
python  ${path}/main.py 2159 &
wait
