import torch
import a301_cc_architecture_4_layer

def load_model_checkpoint(path):
    checkpoint = torch.load(path)

    model = a301_cc_architecture.Classifier(checkpoint["input"])

    model.load_state_dict(checkpoint["state_dict"])

    return model

model = load_model_checkpoint("./checkpoints/checkpoint.pth")

example = torch.tensor([[0.0606, 0.5000, 0.3333, 0.4828, 0.4000, 0.4000, 0.4000, 0.4000, 0.4000,
                         0.4000, 0.1651, 0.0869, 0.0980, 0.1825, 0.1054, 0.2807, 0.0016, 0.0000,
                         0.0033, 0.0027, 0.0031, 0.0021]]).float()

pred = model(example)
pred = torch.exp(pred)
top_p, top_class_test = pred.topk(1, dim=1)
print(top_class_test)