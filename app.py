from flask import Flask,request
from flask_cors import CORS
import numpy as np
import torch
from torch.autograd import Variable
import pickle


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

authKey = "123authKey123"



@app.route('/')
def home():
  return "I like food better than your face"

@app.route('/predict',methods=['POST'])
def predict():
  from rbm  import RBM
  key = request.headers.get('authKey')
  if key != authKey:
    return {
      "msg":"Invalid Access!!!"
    }
  res = request.get_json()
  input_arr = res['input_arr']
  input_arr = torch.FloatTensor(input_arr)
  user_input = Variable(input_arr).unsqueeze(0)
  model = pickle.load(open('models/RBM_Model.pkl','rb'))
  output = model.predict(user_input)
  output = output.data.numpy().tolist()
  return {
    "Output":output
  }

if __name__ == '__main__':
  app.run(debug = True)