


## docker
```bash
dos2unix run.sh
docker build -t repu/torchserve:v1.0.0 .
docker run -dit  -p 8880:8080 -p 8081:8081 --name bert  repu/torchserve:v1.0.0
```