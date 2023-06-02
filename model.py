# In this file we build a simple model with sklearn

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

# Simple data set
data = [
    {"weather" : "sunny", 
     "temperature" : 25, 
     "humidity" : 80, 
     "wind" : 10, 
     "revenue" : "high"},
    {"weather" : "sunny",
        "temperature" : 30,
        "humidity" : 70,
        "wind" : 15,
        "revenue" : "high"},
    {"weather" : "cloudy",
        "temperature" : 22,
        "humidity" : 90,
        "wind" : 5,
        "revenue" : "low"},
    {"weather" : "rainy",
        "temperature" : 18,
        "humidity" : 80,
        "wind" : 20,
        "revenue" : "low"},
    {"weather" : "sunny",
        "temperature" : 20,
        "humidity" : 70,
        "wind" : 28,
        "revenue" : "low"},
    {"weather" : "cloudy",
        "temperature" : 23,
        "humidity" : 80,
        "wind" : 15,
        "revenue" : "high"},
    {"weather" : "rainy",
        "temperature" : 15,
        "humidity" : 80,
        "wind" : 22,
        "revenue" : "low"},
    {"weather" : "cloudy",
        "temperature" : 25,
        "humidity" : 70,
        "wind" : 18,
        "revenue" : "medium"},
    {"weather" : "sunny",
        "temperature" : 12,
        "humidity" : 80,
        "wind" : 25,
        "revenue" : "low"},
    {"weather" : "sunny",
        "temperature" : 24,
        "humidity" : 70,
        "wind" : 10,
        "revenue" : "medium"},
    {"weather" : "rainy",
        "temperature" : 20,
        "humidity" : 80,
        "wind" : 12,
        "revenue" : "medium"},
    {"weather" : "cloudy",
        "temperature" : 9,
        "humidity" : 90,
        "wind" : 8,
        "revenue" : "low"},
    ]

# Convert data to pandas dataframe
df = pd.DataFrame(data)

# Convert categorical data to numerical data
df["weather"] = df["weather"].astype("category").cat.codes
df["revenue"] = df["revenue"].astype("category").cat.codes

# Split data into features and labels
X = df.drop("revenue", axis=1)
y = df["revenue"]

# Build model
model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
model.fit(X, y)

# test model
test_data = [
    {"weather" : "sunny",
        "temperature" : 25,
        "humidity" : 80,
        "wind" : 10},
    {"weather" : "cloudy",
        "temperature" : 25,
        "humidity" : 70,
        "wind" : 18},
    {"weather" : "sunny",
        "temperature" : 12,
        "humidity" : 80,
        "wind" : 25}
    ]
test_labels = ["high", "medium", "low"]

# Convert test data to pandas dataframe
test = pd.DataFrame(test_data)

# Convert categorical data to numerical data
test["weather"] = test["weather"].astype("category").cat.codes

# Print predictions
print("Predictions: ")

print(model.predict(test))

# Save model
dump(model, "model.joblib")


# Train the same model with pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Convert data to tensors
X = torch.tensor(X.values, dtype=torch.float)
y = torch.tensor(y.values, dtype=torch.long)

# Build model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = Net()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train model
for epoch in tqdm(range(10000)):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Test model
test = torch.tensor(test.values, dtype=torch.float)

print("Predictions: ")
print(model(test).argmax(dim=1).numpy())

# Save model
torch.save(model.state_dict(), "model.pt")

