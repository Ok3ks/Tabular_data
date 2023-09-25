FROM PYTHON:3.8.10

RUN pip install -U pip

COPY ["/models", "requirements.txt", "./"]

RUN pipenv install -r requirements.txt

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:96", "predict:app" ]