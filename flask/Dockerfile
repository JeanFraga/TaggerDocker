# FROM python:3.7.6-buster
FROM python:latest

RUN apt-get update \
    && apt-get install -qq -y build-essential libpq-dev python3-setuptools --no-install-recommends

ENV INSTALL_PATH /flask
RUN mkdir $INSTALL_PATH
WORKDIR $INSTALL_PATH

ENV PYTHONUNBUFFERED 1

# WORKDIR /  
# COPY requirements.txt ./  
# RUN pip install --no-cache-dir -r requirements.txt  
# RUN rm requirements.txt

RUN python3 -m pip install --no-binary :all: psycopg2 && \
    python3 -m pip install Flask Flask_SQLAlchemy \
    scikit-learn beautifulsoup4 \
    numpy python_decouple click\
    python_dotenv requests psycopg2-binary \
    lxml pandas basilica gunicorn celery

COPY . .

CMD gunicorn -c "python:config.gunicorn" "tagger:APP"