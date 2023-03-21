for i in {1..13}
do 
echo "CNN_jobs_submitting" ${i}
sbatch jobs/CNN_job_${i}.slurm 
sleep 5
done
