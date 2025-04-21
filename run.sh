#!/bin/sh

cmd="${1:-server}"
if [ "$cmd" = "server" ]; then
   FLASK_APP=serve.py uv run flask run --host 0.0.0.0 --port 8080 --no-debug --with-threads
elif [ "$cmd" = "download" ]; then
    shift
    ./download.sh $@
elif [ "$cmd" = "import" ]; then
    shift
    uv run python generate_db_from_snapshot.py $@
elif [ "$cmd" = "compute_textual" ]; then
    uv run python compute.py
elif [ "$cmd" = "compute_chemical" ]; then
    uv run python decimer.py
elif [ "$cmd" = "compute_img" ]; then
    uv run python img_daemon.py
elif [ "$cmd" = "wait" ]; then
    echo "Waiting infinitely"
    sleep inf
else
    echo "Usage: <server|download|import|compute_textual|compute_chemical|wait>"
    exit 1
fi
