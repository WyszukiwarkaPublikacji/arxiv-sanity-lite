#!/bin/sh

set -e
outdir="${DOWNLOADS_DIR:-$(pwd)/downloads}"
mkdir -p "$outdir"

tmpdir="$(mktemp -d)"
echo "tmpdir: $tmpdir"
cd "$tmpdir"
_cleanup() { rm -rf "$tmpdir"; }
trap _cleanup EXIT

source="$1"
if [ "$source" = "arxiv" ]; then
    uv run kaggle datasets download Cornell-University/arxiv --unzip -f arxiv-metadata-oai-snapshot.json
    mv -v arxiv-metadata-oai-snapshot.json "$outdir/arxiv.json"
elif [ "$source" = "chemrxiv" ]; then
    wget https://github.com/chemrxiv-dashboard/chemrxiv-dashboard.github.io/raw/refs/heads/master/data/allchemrxiv_data.json.bz2
    bzip2 -d allchemrxiv_data.json.bz2
    mv -v allchemrxiv_data.json "$outdir/chemrxiv.json"
else
    echo "Usage: $0 <arxiv|chemrxiv>"
    exit 1
fi
