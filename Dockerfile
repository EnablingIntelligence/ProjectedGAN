FROM python:3.11

WORKDIR /src

# TODO includ model dependencies
COPY ./requirements.txt /src/requirements.txt

RUN pip install --no-chache-dir --upgrade -r /src/requirements.txt

COPY ./web /src/app

# TODO copy model

CMD ["fastapi", "run", "app.py", "--port", "80"]