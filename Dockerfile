FROM nvcr.io/nvidia/pytorch:23.10-py3
WORKDIR /root

RUN apt-get update
RUN mkdir app
COPY ./docker ./app
RUN cd app
RUN pip install git+https://github.com/haven-jeon/PyKoSpacing.git
RUN pip install tensorflow==2.14.0
RUN pip install kss
RUN pip3 install streamlit
RUN pip install datasets transformers

WORKDIR /root/app

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app_roberta.py", "--server.port=8501", "--server.address=0.0.0.0"]
