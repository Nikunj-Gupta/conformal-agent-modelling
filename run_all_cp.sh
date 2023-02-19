list="5 10 15 20 25 30 35 40 45 50"
for cp_update_timestep in $list
do
    mkdir -p out/
    run_cmd="run_cp.sh ${cp_update_timestep}"
    sbatch_cmd="sbatch ${run_cmd}"
    cmd="$sbatch_cmd"
    echo -e "${cmd}"
    ${cmd}
    sleep 1
done