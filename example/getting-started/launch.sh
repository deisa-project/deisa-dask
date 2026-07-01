#!/usr/bin/env bash

export DEISA_DASK_SCHEDULER_ADDRESS="tcp://localhost:8786"

mpirun -np 1 dask scheduler --scheduler-file scheduler.json --protocol tcp --host localhost --port '8786' &
scheduler_pid=$!

sleep 5

mpirun -np 1 dask worker --scheduler-file scheduler.json &
worker_pid=$!

sleep 2

mpirun -np 1 python3 analysis.py &
analysis_pid=$!

mpirun -np 4 python3 simulation.py
simulation_pid=$!

echo "kill worker PID: $worker_pid"
kill $worker_pid
echo "waiting for worker, analysis and simulation PIDs to finish"
wait $worker_pid $analysis_pid $simulation_pid
echo "kill scheduler PID: $scheduler_pid"
kill $scheduler_pid

echo "launcher is done."
