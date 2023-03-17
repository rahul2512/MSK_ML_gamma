##  1270080 total number of hyp
rm jobs/NN_job*slurm
count=0
for (( i=0; i<9721; i+=1 )) 
do
index=$[count/36+1]
count=$[count+1]
echo "python " '${path}'"/main.py" ${i} "&" >> NN_job_${index}.slurm
done 

for index in {1..271}
do 
touch tmp 
cat heading.txt >> tmp
cat NN_job_${index}.slurm >> tmp 
echo "wait" >> tmp
rm NN_job_${index}.slurm
mv tmp jobs/NN_job_${index}.slurm
sed -i "s/RNA9/NN_${index}/g" jobs/NN_job_${index}.slurm
done
