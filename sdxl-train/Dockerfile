# Dockerfile for Diffuser Training.
#
# To build:
# docker build -f model_oss/diffusers/dockerfile/train.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/${YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# Base on pytorch-cuda image.

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        wget \
        git \
        vim

# Install libraries.
RUN pip install torchvision==0.15.2
RUN pip install transformers==4.31.0
RUN pip install accelerate==0.21.0
RUN pip install safetensors==0.3.2
RUN pip install xformers==0.0.21
# RUN pip install bitsandbytes==0.41.1
RUN pip install Jinja2==3.1.2
RUN pip install ftfy==6.1.1
RUN pip install tensorboard==2.14.0
RUN pip install triton==2.0.0.dev20221120
RUN pip install datasets==2.14.4
# RUN pip install cloudml-hypertune==0.1.0.dev6


# Install diffusers from main branch source code with a pinned commit.
RUN git clone --depth 1 --branch main https://github.com/huggingface/diffusers.git
WORKDIR diffusers
RUN pip install -e .

# Switch to diffusers examples folder.
WORKDIR examples

# Config accelerate.
COPY train.sh train.sh

# Generate accelerate config at the beginning of docker run.
ENTRYPOINT ["/bin/bash", "train.sh"]