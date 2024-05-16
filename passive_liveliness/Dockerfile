FROM python:3.10.9

WORKDIR /app

COPY . .

RUN mkdir logs

RUN chmod 0777 -R /app

RUN pip install --upgrade pip setuptools

RUN apt update

RUN apt install -y --fix-missing libopencv-dev python3-opencv

RUN pip install gunicorn numpy

RUN pip install opencv-python==4.5.5.64

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install mtcnn

EXPOSE 5000

#CMD ["sh", "start.sh"]
