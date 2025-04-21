FROM docker.io/library/python:3.11.11-bookworm

RUN apt update && apt install -y libgl1 poppler-utils && apt clean
RUN pip install --no-cache --break-system-packages uv

COPY requirements.txt /root/requirements.txt
RUN uv venv && uv pip install -r /root/requirements.txt

COPY . /app
WORKDIR /app

ENTRYPOINT ["./run.sh"]
