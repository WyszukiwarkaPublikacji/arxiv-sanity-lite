#!/bin/sh
cmd="$1"
if [ "$cmd" = "up" ]; then
    docker compose build
    docker compose up -d
elif [ "$cmd" = "down" ]; then
    docker compose down
elif [ "$cmd" = "logs" ]; then
    shift
    docker compose logs $@
elif [ "$cmd" = "run" ]; then
    shift
    docker compose exec -it worker ./run.sh $@
else
    echo "Usage: $0 <up|down|run> [...]"
    exit 1
fi
