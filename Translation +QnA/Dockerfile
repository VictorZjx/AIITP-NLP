FROM nvcr.io/nvidia/pytorch:22.01-py3
LABEL maintainer="Timothy Liu <timothyl@nvidia.com>"
USER root

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN python3 -m pip install --no-cache-dir -r requirements.txt && \
    rm -rf /tmp/* && \
    rm -rf $HOME/.cache
    
COPY . /workspace/

RUN python cache.py
    
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]

EXPOSE 8501
