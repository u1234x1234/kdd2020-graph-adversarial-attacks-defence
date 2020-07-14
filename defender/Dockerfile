FROM nvidia/cuda:10.2-runtime
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
RUN pip install dgl-cu102
COPY . /app
ENTRYPOINT ["bash","run.sh"]
