FROM nvcr.io/nvidia/tensorflow:22.01-tf2-py3
LABEL maintainer="Timothy Liu <timothyl@nvidia.com>"
USER root

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      graphviz && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /tmp/* && \
    rm -rf $HOME/.cache

COPY requirements.txt /workspace/requirements.txt

RUN python3 -m pip install --no-cache-dir -r requirements.txt && \
    rm -rf /tmp/* && \
    rm -rf $HOME/.cache

EXPOSE 8501
EXPOSE 8888
