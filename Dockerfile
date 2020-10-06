FROM horovod/horovod:latest
RUN pip install -U pip
RUN pip install pytorch-lightning pytorch-lightning-bolts tensorboard
RUN pip install git+https://github.com/kornia/kornia
