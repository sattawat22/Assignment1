import json
import pickle


file_path = "test.json"
with open(file_path, "rb") as file:
    hog = json.load(file)

# print(hog[1])

file_model_path = "model.pkl"
with open(file_model_path, "rb") as file:
    modelRead = pickle.load(file)

result = modelRead.predict([hog[1]])
print(result)
