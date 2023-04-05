#!/bin/bash -l
#SBATCH -J NN_32
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2232 &
python  ${path}/main.py 2233 &
python  ${path}/main.py 2234 &
python  ${path}/main.py 2235 &
python  ${path}/main.py 2236 &
python  ${path}/main.py 2237 &
python  ${path}/main.py 2238 &
python  ${path}/main.py 2239 &
python  ${path}/main.py 2240 &
python  ${path}/main.py 2241 &
python  ${path}/main.py 2242 &
python  ${path}/main.py 2243 &
python  ${path}/main.py 2244 &
python  ${path}/main.py 2245 &
python  ${path}/main.py 2246 &
python  ${path}/main.py 2247 &
python  ${path}/main.py 2248 &
python  ${path}/main.py 2249 &
python  ${path}/main.py 2250 &
python  ${path}/main.py 2251 &
python  ${path}/main.py 2252 &
python  ${path}/main.py 2253 &
python  ${path}/main.py 2254 &
python  ${path}/main.py 2255 &
python  ${path}/main.py 2256 &
python  ${path}/main.py 2257 &
python  ${path}/main.py 2258 &
python  ${path}/main.py 2259 &
python  ${path}/main.py 2260 &
python  ${path}/main.py 2261 &
python  ${path}/main.py 2262 &
python  ${path}/main.py 2263 &
python  ${path}/main.py 2264 &
python  ${path}/main.py 2265 &
python  ${path}/main.py 2266 &
python  ${path}/main.py 2267 &
python  ${path}/main.py 2268 &
python  ${path}/main.py 2269 &
python  ${path}/main.py 2270 &
python  ${path}/main.py 2271 &
python  ${path}/main.py 2272 &
python  ${path}/main.py 2273 &
python  ${path}/main.py 2274 &
python  ${path}/main.py 2275 &
python  ${path}/main.py 2276 &
python  ${path}/main.py 2277 &
python  ${path}/main.py 2278 &
python  ${path}/main.py 2279 &
python  ${path}/main.py 2280 &
python  ${path}/main.py 2281 &
python  ${path}/main.py 2282 &
python  ${path}/main.py 2283 &
python  ${path}/main.py 2284 &
python  ${path}/main.py 2285 &
python  ${path}/main.py 2286 &
python  ${path}/main.py 2287 &
python  ${path}/main.py 2288 &
python  ${path}/main.py 2289 &
python  ${path}/main.py 2290 &
python  ${path}/main.py 2291 &
python  ${path}/main.py 2292 &
python  ${path}/main.py 2293 &
python  ${path}/main.py 2294 &
python  ${path}/main.py 2295 &
python  ${path}/main.py 2296 &
python  ${path}/main.py 2297 &
python  ${path}/main.py 2298 &
python  ${path}/main.py 2299 &
python  ${path}/main.py 2300 &
python  ${path}/main.py 2301 &
python  ${path}/main.py 2302 &
python  ${path}/main.py 2303 &
wait
