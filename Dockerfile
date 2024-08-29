FROM python:3.11

WORKDIR /src/app

# TODO includ model dependencies
COPY ./requirements.txt /src/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /src/requirements.txt

COPY ./app .

# TODO copy model

EXPOSE 8080
CMD ["fastapi", "run", "main.py", "--port", "8080"]