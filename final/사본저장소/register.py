import requests

url = "http://10.30.117.40:5000/register"

response = requests.post(url, json={"email":"1234", "password": "123" , "name": "moon"})

print(response.content)