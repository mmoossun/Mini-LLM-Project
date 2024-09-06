import requests

url = "http://10.30.117.40:5000/login"

response = requests.post(url, json={"email":"1234", "password": "123"})

print(response.content)