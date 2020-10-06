FROM idein/pytorch:latest
RUN pip install -U pip
RUN pip install pytorch-lightning pytorch-lightning-bolts tensorboard==2.2.0 sklearn
RUN pip install git+https://github.com/kornia/kornia
