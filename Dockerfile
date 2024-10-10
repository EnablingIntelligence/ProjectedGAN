FROM python:3.11

WORKDIR /src/app

COPY ./web_requirements.txt /src/web_requirements.txt
RUN pip install --no-cache-dir --upgrade -r /src/web_requirements.txt

COPY ./app .

COPY ./model_requirements.txt /src/model_requirements.txt
RUN pip install --no-cache-dir --upgrade -r /src/model_requirements.txt

COPY ./gan .

EXPOSE 8080
CMD ["fastapi", "run", "main.py", "--port", "8080"]