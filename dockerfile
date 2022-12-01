# The CNETS-info image creation
FROM python:3.9-slim-bullseye
WORKDIR .
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir  --requirement /tmp/requirements.txt
RUN pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install sklearn
COPY ./app.py /
COPY ./converted_model.tflite /
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
 