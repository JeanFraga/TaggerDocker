echo "Using echo command to describe what is going on in this file"

echo "This lines of code will run the commands that upgrade the ubuntu AMI to run the latest packages"
echo "-y confirms any prompts that the machine might ask for in user input"
sudo apt install git-all
sudo apt-get update && sudo apt-get upgrade -y

echo "This line installs docker with sudo"
sudo apt install docker.io -y

echo "This line installs docker-compose with sudo"
curl -L https://github.com/docker/compose/releases/download/1.25.4/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

echo "This line clones the repo that contains the docker compose and flask"
git clone https://github.com/JeanFraga/TaggerDocker
cd TaggerDocker

echo "this line will return an error if you run without .env but I left it here for show and tell"
sudo docker-compose up --build
