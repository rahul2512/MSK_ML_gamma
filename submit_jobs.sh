for i in {1..288}
do 
echo "RNN_jobs_submitting" ${i}
sbatch jobs/RNN_job_${i}.slurm 
sleep 100
done
