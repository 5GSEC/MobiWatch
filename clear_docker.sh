sudo docker container prune
#docker rmi $(docker images -q -f "dangling=true" -f "label=autodelete=true")
sudo docker rmi $(docker images -f "dangling=true" -q)
