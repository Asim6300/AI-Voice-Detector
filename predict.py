import joblib
from feature_extractor import extract_features

clf=joblib.load("models/classifier.pkl")
scaler=joblib.load("models/scaler.pkl")

def predict(path):
    x=extract_features(path)
    x=scaler.transform([x])
    p=clf.predict_proba(x)[0]
    human=p[0]*100
    ai=p[1]*100
    result="Uncertain" if abs(human-ai)<10 else ("Human" if human>ai else "AI")
    return {"result":result,"human":round(human,2),"ai":round(ai,2)}
