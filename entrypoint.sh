#!/bin/bash

cmd="$1"
shift
if [ -z "$cmd" ]; then
    cmd="server"
fi

if [ "$cmd" == "server" ]; then
   FLASK_APP=serve.py uv run flask run $@
elif [ "$cmd" == "import_snapshot" ]; then
    uv run python generate_db_from_snapshot.py $@
elif [ "$cmd" == "compute_features" ]; then
    uv run python compute.py $@
elif [ "$cmd" == "decimer" ]; then
    uv run python decimer.py $@
else
    echo "Invalid command."
    exit 1
fi
