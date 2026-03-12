#!/bin/bash

#SBATCH --job-name=deisa_scaling
#SBATCH --output=%x_%j.out
#SBATCH --time=00:20:00
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=20
#SBATCH --threads-per-core=1
#SBATCH --partition=cpu_short

SCHEFILE=/home/bemartin/storage/numpexexadi/deisa-bench/scheduler.json
DASK_WORKER_NODES=1
DASK_NB_WORKERS=1                # Number of Dask workers
DASK_NB_THREAD_PER_WORKER=0      # Number of threads per Dask workers
NB_CLIENTS=(8 16 32)
DATA_SIZES=(64 128 256)
NB_RUNS=5

# Launch Dask Scheduler in a 1 Node and save the connection information in $SCHEFILE
echo launching Scheduler
srun -N 1 -n 1 -c 1 -r 0 dask scheduler --scheduler-file=$SCHEFILE &
dask_sch_pid=$!

# Wait for the SCHEFILE to be created
while ! [ -f $SCHEFILE ]; do
    sleep 1
    echo -n .
done

echo Scheduler booted, launching workers
srun -N ${DASK_WORKER_NODES} -n ${DASK_WORKER_NODES} -c 1 -r 1 dask worker \
    --nworkers ${DASK_NB_WORKERS} \
    --nthreads ${DASK_NB_THREAD_PER_WORKER} \
    --local-directory /tmp \
    --scheduler-file=${SCHEFILE} &
dask_worker_pid=$!

sleep 1

for data_size in "${DATA_SIZES[@]}"; do
  for nb_clients in "${NB_CLIENTS[@]}"; do
    for run_id in $(seq 1 ${NB_RUNS}); do

      for id in $(seq 1 "${nb_clients}"); do
        # scheduler file, nb clients, data size, run_id, unique id
        srun -N 1 -n 1 -r $(($DASK_WORKER_NODES+2)) \
          python3 ./connect-clients.py $SCHEFILE "${nb_clients}" "${data_size}" "${run_id}" "${id}" &
        pids+=($!)
      done

      wait "${pids[@]}" # wait for all background tasks
      sleep 5
    done
  done
done

sleep 1

kill -9 ${dask_worker_pid} ${dask_sch_pid}
rm $SCHEFILE

#sleep 99999999999

echo "done"