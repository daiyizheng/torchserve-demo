


## docker
```bash
dos2unix run.sh
docker build -t repu/torchserve:v1.0.0 .
docker run -dit  -p 8880:8080 -p 8081:8081 --name bert  repu/torchserve:v1.0.0
```

## 接口调用
```python
import requests
import json
def tester():
    server_url = 'http://xxxx:8880/predictions/bertEmbbeding'

    payload = {
        "text":["做基因检测采集什么样本检测结果最准确"],
    }
    json_payload = json.dumps(payload)
    content_length = len(json_payload)

    headers = {'Content-Type': 'application/text', 'Content-Length': '2'}
    response = requests.post(server_url, data=json_payload, headers=headers, allow_redirects=True)

    if response.status_code == requests.codes.ok:
        print ('Headers: {}\nResponse: {}'.format(response.headers, response.text))
```