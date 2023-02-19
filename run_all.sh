envs="simple_spread_v2 simple_push_v2 simple_speaker_listener_v3 simple_world_comm_v2"
baselines="giam noam"
for env in $envs
do
    for baseline in $baselines
        do
            mkdir -p out/
            run_cmd="run_baselines.sh ${env} ${baseline}"
            sbatch_cmd="sbatch ${run_cmd}"
            cmd="$sbatch_cmd"
            echo -e "${cmd}"
            ${cmd}
            sleep 1
        done
done