from flask import Flask,request
from flask_cors import CORS
import numpy as np
import torch
from torch.autograd import Variable
import pickle

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

authKey = "123authKey123"


class RBM():
  def __init__(self, nv, nh):
    self.W = torch.randn(nh, nv)
    self.a = torch.randn(1, nh)
    self.b = torch.randn(1, nv)

  def sample_h(self, x):
    wx = torch.mm(x, self.W.t())
    activation = wx + self.a.expand_as(wx)
    p_h_given_v = torch.sigmoid(activation)
    return p_h_given_v, torch.bernoulli(p_h_given_v)

  def sample_v(self, y):
    wy = torch.mm(y, self.W)
    activation = wy + self.b.expand_as(wy)
    p_v_given_h = torch.sigmoid(activation)
    return p_v_given_h, torch.bernoulli(p_v_given_h)

  def train(self, v0, vk, ph0, phk):
    self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t()
    self.b += torch.sum((v0 - vk), 0)
    self.a += torch.sum((ph0 - phk), 0)
    
  def predict(self, x):
    _, h = self.sample_h(x)
    _, v = self.sample_v(h)
    return v

@app.route('/')
def home():
  return "I like food better than your face"

@app.route('/predict')
def predict():
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