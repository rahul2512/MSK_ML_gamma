for i in {2..271}
do 
echo "Submitting job ${i}"
sbatch jobs/NN_job_${i}.slurm 
sleep 65
done
