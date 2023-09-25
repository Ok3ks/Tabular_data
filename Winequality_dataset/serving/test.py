import requests


#should probably add a vectorizer for features 
wine = {"sulphates": 4,
    "chlorides": 5,
    "volatile acidity": 6,
    "quality": 7,
    "alcohol": 8,
    "pH": 9, 
    "residual sugar": 10,
    "total sulfur dioxide":12}


url = "http://localhost:9696/predict"
response = requests.post(url, json=wine)
print(response.json())