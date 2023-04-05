#!/bin/bash -l
#SBATCH -J NN_28
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
python  ${path}/main.py 1980 &
python  ${path}/main.py 1981 &
python  ${path}/main.py 1982 &
python  ${path}/main.py 1983 &
python  ${path}/main.py 1984 &
python  ${path}/main.py 1985 &
python  ${path}/main.py 1986 &
python  ${path}/main.py 1987 &
python  ${path}/main.py 1988 &
python  ${path}/main.py 1989 &
python  ${path}/main.py 1990 &
python  ${path}/main.py 1991 &
python  ${path}/main.py 1992 &
python  ${path}/main.py 1993 &
python  ${path}/main.py 1994 &
python  ${path}/main.py 1995 &
python  ${path}/main.py 1996 &
python  ${path}/main.py 1997 &
python  ${path}/main.py 1998 &
python  ${path}/main.py 1999 &
python  ${path}/main.py 2000 &
python  ${path}/main.py 2001 &
python  ${path}/main.py 2002 &
python  ${path}/main.py 2003 &
python  ${path}/main.py 2004 &
python  ${path}/main.py 2005 &
python  ${path}/main.py 2006 &
python  ${path}/main.py 2007 &
python  ${path}/main.py 2008 &
python  ${path}/main.py 2009 &
python  ${path}/main.py 2010 &
python  ${path}/main.py 2011 &
python  ${path}/main.py 2012 &
python  ${path}/main.py 2013 &
python  ${path}/main.py 2014 &
python  ${path}/main.py 2015 &
wait
