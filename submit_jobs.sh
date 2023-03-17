for i in {1..1216}
do 
echo "Submitting job ${i}"
sbatch jobs/job_${i}.slurm 
sleep 65
done
