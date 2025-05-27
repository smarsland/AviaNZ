from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import torchinfo
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from NNModels import PatchLayer, PositionalEmbedding, TransformerBlock, AudioSpectogramTransformer
import torch

# The following is code to convert a ViT transformer from pytorch to tensorflow. 
# Pytorch puts the outputs first, so a lot of weights need to be transposed. 

model1 = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k',output_hidden_states=True)
model2 = AudioSpectogramTransformer(224, 224, 0) # no head

# # print the parameters and their sizes in model1
# print("Model 1 Parameters:")
# for name, param in model1.named_parameters():
#     print(name, param.size())

# # print the parameters and their sizes in model2
# print("Model 2 Parameters:")
# for layer in model2.layers:
#     for weight in layer.weights:
#         print(weight.name, weight.shape)

model2.get_layer("cls_token_layer").set_weights([model1.embeddings.cls_token.detach().numpy()])
model2.get_layer("positional_embedding").set_weights([model1.embeddings.position_embeddings.detach().numpy()])
embeddingWeight = model1.embeddings.patch_embeddings.projection.weight.detach().numpy()
embeddingWeight = embeddingWeight.transpose(0,2,3,1).reshape(768,-1).T # Move the channels to the end, combine inputs.
embeddingBias = model1.embeddings.patch_embeddings.projection.bias.detach().numpy()
model2.get_layer("time_distributed").set_weights([embeddingWeight, embeddingBias])
for l in range(1):
    transformerWeights = [
        model1.encoder.layer[l].layernorm_before.weight.detach().numpy(),
        model1.encoder.layer[l].layernorm_before.bias.detach().numpy(),
        model1.encoder.layer[l].attention.attention.query.weight.detach().numpy().T.reshape(768, 12, 64),
        model1.encoder.layer[l].attention.attention.query.bias.detach().numpy().reshape(12, 64),
        model1.encoder.layer[l].attention.attention.key.weight.detach().numpy().T.reshape(768, 12, 64),
        model1.encoder.layer[l].attention.attention.key.bias.detach().numpy().reshape(12, 64),
        model1.encoder.layer[l].attention.attention.value.weight.detach().numpy().T.reshape(768, 12, 64),
        model1.encoder.layer[l].attention.attention.value.bias.detach().numpy().reshape(12, 64),
        model1.encoder.layer[l].attention.output.dense.weight.detach().numpy().T.reshape(12, 64, 768),
        model1.encoder.layer[l].attention.output.dense.bias.detach().numpy().reshape(768),
        model1.encoder.layer[l].layernorm_after.weight.detach().numpy(),
        model1.encoder.layer[l].layernorm_after.bias.detach().numpy(),
        model1.encoder.layer[l].intermediate.dense.weight.detach().numpy().T,
        model1.encoder.layer[l].intermediate.dense.bias.detach().numpy(),
        model1.encoder.layer[l].output.dense.weight.detach().numpy().T,
        model1.encoder.layer[l].output.dense.bias.detach().numpy()
    ]
    name = "transformer_block_" + str(l) if l>0 else "transformer_block"
    model2.get_layer(name).set_weights(transformerWeights)

# save model 2
model2.save_weights("pre-trained_ViT.weights.h5")