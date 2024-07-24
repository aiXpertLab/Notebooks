# {
#     "data": [
#         [0.0606, 0.5000, 0.3333, 0.4828, 0.4000, 0.4000, 0.4000, 0.4000, 0.4000,
#          0.4000, 0.1651, 0.0869, 0.0980, 0.1825, 0.1054, 0.2807, 0.0016, 0.0000,
#          0.0033, 0.0027, 0.0031, 0.0021]
#     ]
# }


import flask
from flask import request
import torch
import a303_cc_final_model
app = flask.Flask(__name__)
app.config["DEBUG"] = True

def load_model_checkpoint(path):
    checkpoint = torch.load(path)

    model = a303_cc_final_model.Classifier(checkpoint["input"])

    model.load_state_dict(checkpoint["state_dict"])

    return model

model = load_model_checkpoint("../checkpoints/checkpoint_303_f.pth")
@app.route('/prediction', methods=['POST'])
def prediction():
    model.eval()
    body = request.get_json()

    example = torch.tensor(body['data']).float()

    pred = model(example)
    pred = torch.exp(pred)
    _, top_class_test = pred.topk(1, dim=1)
    top_class_test = top_class_test.numpy()

    return {"status":"ok", "result":int(top_class_test[0][0])}
app.run(debug=True, use_reloader=False)


