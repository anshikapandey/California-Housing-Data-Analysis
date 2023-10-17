from flask import Flask
import joblib

app = Flask(__name__)

# Load your pre-trained model
model = joblib.load('California_Housing_Data_Analysis.ipynb')

@app.route('/')
def home():
    # Perform some predictions with your model
    prediction = model.predict([[1.000000, -0.924664, -0.108197, 0.044568, 0.069608, 0.099773, 0.055310, -0.015176]])  # Adjust the input as per your model's requirements
    return f'The prediction is {prediction}'

if __name__ == '__main__':
    app.run(debug=True)
