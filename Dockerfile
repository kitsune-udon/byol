FROM horovod/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.6.0.post0-py3.7-cuda10.1
RUN pip install -U pip
RUN pip install pytorch-lightning pytorch-lightning-bolts tensorboard
RUN pip install git+https://github.com/kornia/kornia
