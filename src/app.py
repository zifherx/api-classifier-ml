from flask import Flask,jsonify,request,Response
from flask.helpers import send_from_directory
import joblib
import pickle
import numpy as np
from transformers import BertModel, BertTokenizer,AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn,optim
from flask_swagger_ui import get_swaggerui_blueprint
from urllib.request import urlopen

app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SextingSentimentClassifier(nn.Module):
  def __init__(self,n_classes):
    super(SextingSentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)
    self.drop = nn.Dropout(p=0.3)
    self.linear = nn.Linear(self.bert.config.hidden_size, n_classes) #Creando capa adicional de una red neuronal

  def forward(self, input_ids, attention_mask):
    _, cls_output = self.bert(
        input_ids = input_ids,
        attention_mask = attention_mask
    )
    drop_output = self.drop(cls_output)
    output = self.linear(drop_output)
    return output

url = "https://dl.dropboxusercontent.com/s/f0aajsnkijtsrhk/modelov7_cpu.pkl"
#modelo_prueba = joblib.load(urlopen(url))

SEXTING_MODEL = joblib.load(urlopen(url))

PRE_TRAINED_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

print('Modelo cargado exitosamente.')

def clasificadordelSexting(frase_text):
  encoding_frase = tokenizer.encode_plus(
      frase_text,
      max_length = 15,
      truncation = True,
      add_special_tokens = True,
      return_token_type_ids = False,
      return_attention_mask = True,
      return_tensors = 'pt'
  )
  input_ids = encoding_frase['input_ids'].to(device)
  attention_mask = encoding_frase['attention_mask'].to(device)
  output = SEXTING_MODEL(input_ids, attention_mask)
  _, prediction = torch.max(output, dim=1)
  #print("\n".join(wrap(frase_text)))
  #print(output)
  #print(prediction)
  if prediction:
    return 'No Sexting'
  else:
    return 'Sexting'

@app.route('/', methods=['GET','POST'])
def predecir():
    oracion = request.json['oracion']
    if oracion:
        response = { 'Clasificacion' : clasificadordelSexting(oracion)}
        return response
    else:
        return not_found()

@app.route('/static/<path:path>')
def send_static(path):
  return send_from_directory('static', path)

SWAGGER_URL = '/docs'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
  SWAGGER_URL,
  API_URL,
  config={
    'app_name' : 'Sexting Scan API',
  }
)
app.register_blueprint(swaggerui_blueprint, url_prefix = SWAGGER_URL)

@app.errorhandler(404)
def not_found(error=None):

    response = jsonify({
        'message' : 'Resource Not Found - ' + request.url,
        'status': 404
    })
    response.status_code = 404
    return response

#Listener
if __name__ == '__main__':
    app.run(debug=True)