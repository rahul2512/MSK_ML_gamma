##  1270080 total number of hyp
rm jobs/CNN_job*slurm
count=0
for (( i=0; i<865; i+=1 )) 
do
index=$[count/24+1]
count=$[count+1]
echo "python " '${path}'"/main.py" ${i} "&" >> CNN_job_${index}.slurm
done 

for index in {1..37}
do 
touch tmp 
cat heading.txt >> tmp
cat CNN_job_${index}.slurm >> tmp 
echo "wait" >> tmp
rm CNN_job_${index}.slurm
mv tmp jobs/CNN_job_${index}.slurm
sed -i "s/MMM/CNN_${index}/g" jobs/CNN_job_${index}.slurm
done
