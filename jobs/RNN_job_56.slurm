#!/bin/bash -l
#SBATCH -J RNN_56
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
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
