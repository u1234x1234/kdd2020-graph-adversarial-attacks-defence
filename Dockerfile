FROM nvidia/cuda:10.2-runtime
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip
WORKDIR /app
COPY requirements.txt /app
RUN pip install dgl-cu102==0.5.0

ENV CUDA cu102
RUN pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
    torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
    torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
    torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
    torch-geometric==1.6.0

COPY . /app
ENV PYTHONPATH /app/defender
ENTRYPOINT ["python3","generate_solution.py"]
