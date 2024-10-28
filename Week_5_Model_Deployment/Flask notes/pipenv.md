pip install pipenv
pipenv install numpy pandas scikit-sklearn waitress 

# activate the shell 
pipenv shell 

# see the location of env 
ls
ls source /Users/sudhirjoon/.local/share/virtualenvs/Week_5_Model_Deployment-4SZwTQSZ/bin  
which waitress-serve ... it will tell you which env you are using
echo $path (the first path of environment is attached to our main directiry)

# remove all the docker images
docker ps # list all running containers 
docker stop container_id
docker rmi $(docker images -q) --force 