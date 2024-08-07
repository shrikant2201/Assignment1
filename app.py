### Flask application to serve the model 
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify(prediction=prediction.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
