for i in {1..145}
do 
echo "Submitting job ${i}"
sbatch jobs/NN_job_${i}.slurm 
sleep 20
done
