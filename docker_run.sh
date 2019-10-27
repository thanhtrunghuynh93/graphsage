#docker build -t pytorch .
docker run --runtime=nvidia -it -v $HOME:/app pytorch /bin/bash 
#nvidia-docker run -it -v /home/bigdata/thomas:/home tensorflow/tensorflow:1.8.0-gpu bash
