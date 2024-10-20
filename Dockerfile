FROM python:3.11

WORKDIR /src

RUN apt update
RUN apt -y install libmpc-dev

COPY ./web_requirements.txt /src/web_requirements.txt
RUN pip install --no-cache-dir --upgrade -r /src/web_requirements.txt

COPY ./model_requirements.txt /src/model_requirements.txt
RUN pip install --no-cache-dir --upgrade -r /src/model_requirements.txt

COPY ./app ./app
COPY ./gan ./gan

EXPOSE 8080
CMD ["fastapi", "run", "./app/main.py", "--port", "8080"]