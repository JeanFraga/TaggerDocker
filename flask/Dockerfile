FROM python:3.7.6-buster

RUN apt-get update \
    && apt-get install -qq -y build-essential libpq-dev python3-setuptools --no-install-recommends

ENV INSTALL_PATH /flask
RUN mkdir $INSTALL_PATH
WORKDIR $INSTALL_PATH

ENV PYTHONUNBUFFERED 1

RUN python3 -m pip install --no-binary :all: psycopg2 && \
    python3 -m pip install Flask==1.1.1 Flask_SQLAlchemy==2.4.1 \
    scikit-learn beautifulsoup4==4.8.1 \
    numpy==1.17.4 python_decouple==3.3 click\
    python_dotenv==0.10.3 requests==2.22.0 psycopg2-binary==2.8.4 \
    lxml==4.4.2 pandas==0.25.3 basilica gunicorn

COPY . .

CMD gunicorn -c "python:config.gunicorn" "tagger:APP"