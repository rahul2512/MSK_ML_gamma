for i in {1..30}
do 
echo "rf_jobs_submitting" ${i}
sbatch jobs/rf_job_${i}.slurm 
sleep 25
done
