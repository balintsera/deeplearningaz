FROM continuumio/anaconda3:4.4.0

RUN pip install keras tensorflow
RUN conda update libgcc -y
RUN apt-get update && apt-get install libgl1-mesa-glx libgcc libgomp1 -y

ENTRYPOINT [ "/bin/bash", "-c", "/opt/conda/bin/conda install jupyter -y --quiet && /opt/conda/bin/jupyter notebook --notebook-dir=/work/ --ip='*' --port=8867 --no-browser --allow-root" ]