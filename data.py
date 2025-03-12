import pandas as pd
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=2,
n_classes=2, n_informative=2,
n_redundant=0, random_state=42)
df = pd.DataFrame(X, columns=["feature1", "feature2"])
df["target"] = y
df.to_csv("data.csv", index=False)
import pandas as pd
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=2,
n_classes=2, n_informative=2,
n_redundant=0, random_state=42)
df = pd.DataFrame(X, columns=["feature1", "feature2"])
df["target"] = y
df.to_csv("data.csv", index=False)
