import os
import joblib
import numpy as np
from feature_extractor import extract_features
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X=[]
Y=[]

for cls in ["human","ai"]:

    folder=f"datasets/{cls}"

    files=os.listdir(folder)

    print(f"\nLoading {cls}: {len(files)} files")

    for f in files:

        path=os.path.join(folder,f)

        try:
            feat=extract_features(path)

            X.append(feat)

            Y.append(0 if cls=="human" else 1)

            print("Loaded:",f)

        except Exception as e:

            print("Skipped:",f)
            print(e)

X=np.array(X)
Y=np.array(Y)

print("\nTotal samples:",len(X))

if len(X)==0:
    raise Exception("No audio loaded")

scaler=StandardScaler()

X=scaler.fit_transform(X)

os.makedirs("models",exist_ok=True)

joblib.dump(
scaler,
"models/scaler.pkl"
)

xtr,xte,ytr,yte=train_test_split(
X,
Y,
test_size=.2,
stratify=Y,
random_state=42
)

print("\nTraining model...")

clf=RandomForestClassifier(
n_estimators=300,
class_weight="balanced",
random_state=42
)

clf.fit(xtr,ytr)

acc=clf.score(xte,yte)

print("\nAccuracy:",acc)

joblib.dump(
clf,
"models/classifier.pkl"
)

print("\nTraining complete")
print("Saved:")
print("models/classifier.pkl")
print("models/scaler.pkl")