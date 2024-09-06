import requests

url = "http://10.30.117.40:5000/name"

response = requests.get(url, json={"token":"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6IjEyMzQiLCJleHAiOjE3MjE4MzM5MDJ9.vbBPLyVi0eNes1HyWziVJX0MD6jDePQGtQ7gt-oWIUo"})

print(response.content)