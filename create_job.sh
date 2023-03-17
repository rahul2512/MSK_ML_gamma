##  1270080 total number of hyp
rm jobs/job*slurm
count=0
for (( i=0; i<43741; i+=1 )) 
do
index=$[count/36+1]
count=$[count+1]
echo "python " '${path}'"/main.py" ${i} "&" >> job_${index}.slurm
done 

for index in {1..1216}
do 
touch tmp 
cat heading  >>  tmp
cat job_${index}.slurm >> tmp 
echo "wait" >> tmp
rm job_${index}.slurm
mv tmp jobs/job_${index}.slurm
sed -i "s/RNA9/job_${index}/g" jobs/job_${index}.slurm
done
