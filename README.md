
# arxiv-sanity-lite

A much lighter-weight arxiv-sanity from-scratch re-write. Periodically polls arxiv API for new papers. Then allows users to tag papers of interest, and recommends new papers for each tag based on SVMs over tfidf features of paper abstracts. Allows one to search, rank, sort, slice and dice these results in a pretty web UI. Lastly, arxiv-sanity-lite can send you daily emails with recommendations of new papers based on your tags. Curate your tags, track recent papers in your area, and don't miss out!

I am running a live version of this code on [arxiv-sanity-lite.com](https://arxiv-sanity-lite.com).

![Screenshot](screenshot.jpg)

## Running (Docker)
```sh
# Start the stack
sudo ./eztoolbox.sh up

# Download snapshots
sudo ./eztoolbox.sh run download chemrxiv
sudo ./eztoolbox.sh run download arxiv

# Import snapshots
sudo ./eztoolbox.sh run import -c -f /downloads/chemrxiv.json
sudo ./eztoolbox.sh run import -a -f /downloads/arxiv.json

# Compute textual features
sudo ./eztoolbox.sh run compute_textual

# Compute chemical features (structures detection and fingerprinting). Takes a lot time, but Ctrl+C is your friend
sudo ./eztoolbox.sh run compute_chemical  # 

# Stop and remove the stack
sudo ./eztoolbox.sh down
```

## Running (old)
for chemrxiv we use snapshots from the [chemrxiv-dashboard](https://github.com/chemrxiv-dashboard/chemrxiv-dashboard.github.io) project, and run the respective script.
```bash
python3 generate_db_from_chemrxiv.py -f allchemrxiv_data.json
```
We also need to extract SMILES from the papers(keep in mind it is an extremely long process that hasn't yet ever been fully completed).
To do that we run the decimer.py script.
```bash
python3 decimer.py
```
However it only seems to work on older python versions. We had success running it on a dockerized debian 11, or under uv.
Additionally it requries poppler, which on debian based system can by installed by running
```bash
sudo apt install poppler-utils
```
Finally to serve the flask server locally we'd run something like:

On Windows sometimes it is better to run this line before:
```bash
$env:FLASK_APP="serve.py"
```
Also, if you want to use milvus standlone hosted on http://localhost:19530/ set:
```bash
$env:MILVUS_MODE="standalone"
```
Otherwise it will try to use milvus lite with database stored locally.

```bash
export FLASK_APP=serve.py; flask run
```

All of the database will be stored inside the `data` directory. Finally, if you'd like to run your own instance on the interwebs I recommend simply running the above on a [Linode](https://www.linode.com), e.g. I am running this code currently on the smallest "Nanode 1 GB" instance indexing about 30K papers, which costs $5/month.

(Optional) Finally, if you'd like to send periodic emails to users about new papers, see the `send_emails.py` script. You'll also have to `pip install sendgrid`. I run this script in a daily cron job.

#### Requirements

 Install via requirements:

 ```bash
 pip install -r requirements.txt
 ```

#### Todos

- Make website mobile friendly with media queries in css etc
- The metas table should not be a sqlitedict but a proper sqlite table, for efficiency
- Build a reverse index to support faster search, right now we iterate through the entire database

#### License

MIT
