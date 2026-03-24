from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_iris()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2
)

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

print("Training completed successfully")