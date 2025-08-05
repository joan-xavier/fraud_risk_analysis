import pandas as pd

df = pd.read_csv("AIML_Dataset.csv")
df.to_pickle("AIML_Dataset.pkl")

# Re-train and export in your current environment
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import pandas as pd

df = pd.read_pickle("AIML_Dataset.pkl")
X = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
y = df['isFraud']

# OneHot + StandardScaler
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['type'])
])

# Logistic Regression model
logreg_pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000))
])
logreg_pipe.fit(X, y)
joblib.dump(logreg_pipe, "fraud_model_logreg.pkl")

# Decision Tree model
tree_pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', DecisionTreeClassifier(random_state=42))
])
tree_pipe.fit(X, y)
joblib.dump(tree_pipe, "fraud_model_tree.pkl")
