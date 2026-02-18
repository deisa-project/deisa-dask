#!/bin/bash
SCHEFILE=scheduler.json
DASK_NB_WORKERS=1                # Number of Dask workers
DASK_NB_THREAD_PER_WORKER=0      # Number of threads per Dask workers
NB_CLIENTS=(8 16 32)
DATA_SIZES=(64 128 256)
NB_RUNS=5



# Launch Dask Scheduler in a 1 Node and save the connection information in $SCHEFILE
echo launching Scheduler
dask scheduler \
    --interface lo \
    --scheduler-file=$SCHEFILE &
dask_sch_pid=$!

# Wait for the SCHEFILE to be created
while ! [ -f $SCHEFILE ]; do
    sleep 1
    echo -n .
done

echo Scheduler booted, launching workers
dask worker \
    --interface lo \
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
        python3 ./connect-clients.py $SCHEFILE "${nb_clients}" "${data_size}" "${run_id}" "${id}" &
        pids+=($!)
      done

      wait "${pids[@]}" # wait for all background tasks
      sleep 5
    done
  done
done

#python3 ./connect-clients.py $SCHEFILE 10 ${DATA_SIZE}

sleep 1

kill -9 ${dask_worker_pid} ${dask_sch_pid}
rm $SCHEFILE

echo "done"