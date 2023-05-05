from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.nn.functional as F
import torch
from configs import *
from transformers import activations
from activations.prelu import *

def convert_act_class(model, layer_type_old, layer_type_new):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        #print("In : ", name, module)
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_act_class(module, layer_type_old, layer_type_new)

        if type(module) == layer_type_old:
            #print("Chnaging", model._modules[name])
            layer_old = module
            layer_new = layer_type_new
            model._modules[name] = layer_new

    return model


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
bert_model = SentenceTransformer(bert_model_name).to(DEVICE)

print(bert_model[0]._modules.items())
convert_act_class(bert_model, activations.GELUActivation, PReLU(alpha = 0.8))
print(bert_model[0]._modules.items())

sent = ["Hello there."]

emb = bert_model.encode(sent)
print(emb)