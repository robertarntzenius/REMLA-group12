FROM python:3.7.10-slim

RUN apt-get clean \
&& apt-get update \
&& apt-get install -y --no-install-recommends git \
&& apt-get purge -y --auto-remove \
&& apt-get -y install nginx \
   python3-dev \
   build-essential \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /root/

ENV VIRTUAL_ENV=/root/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .

RUN python -m pip install --upgrade pip &&\
    pip install -r requirements.txt

COPY main.py .
COPY flaskapi.py .
COPY src src
COPY data data
COPY reports reports

EXPOSE 8080

CMD [ "python", "flaskapi.py" ]
