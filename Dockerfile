FROM tensorflow/tensorflow:2.1.1-gpu

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /opt/DLHomework

COPY requirements.txt .
#RUN apt update && apt install python3-pip -y
RUN pip3 install --no-cache-dir -r requirements.txt

#ADD . /opt/DLHomework