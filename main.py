from flask import Flask, jsonify, request
import pickle
from Controllers import transformText
from flask_cors import CORS
# from waitress import serve
app = Flask(__name__)

# Handling CORS
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# importing models
model = pickle.load(open('./model/model.pkl', 'rb'))
tfidf = pickle.load(open('./model/vectorizer.pkl', 'rb'))

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/hello')
def hey():
    return 'Hello, Jaydeep!'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    transformed_sms = transformText.transform_text(data['sentence'])
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    return jsonify({'result': int(result)})
    # print(data)
    # return jsonify(data)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
    # serve(app, port=5000)