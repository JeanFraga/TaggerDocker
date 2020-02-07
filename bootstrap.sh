sudo apt install git-all
sudo apt-get update && sudo apt-get upgrade -y

curl -L https://github.com/docker/compose/releases/download/1.25.4/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

git clone --single-branch --branch Jean_EC2test https://github.com/JeanFraga/TaggerDocker
cd taggerDocker

sudo docker-compose up --build