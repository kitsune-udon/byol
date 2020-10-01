FROM idein/pytorch:latest
RUN pip install pytorch-lightning pytorch-lightning-bolts tensorboard
RUN pip install git+https://github.com/kornia/kornia