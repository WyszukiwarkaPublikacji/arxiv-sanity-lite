FROM docker.io/library/python:3.11.11-bookworm

RUN apt update && apt install -y libgl1 poppler-utils && apt clean
RUN pip install --no-cache --break-system-packages uv

COPY . /app
WORKDIR /app

RUN uv venv && uv pip install -r requirements.txt

ENTRYPOINT ["./entrypoint.sh"]
