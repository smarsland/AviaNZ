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

modelName = "WinKawaks/vit-tiny-patch16-224"

if modelName == "google/vit-base-patch16-224-in21k":
    transformerHeads = 12
    transformerLayers = 12
    embeddingDim = 768
    transformerNNDim = 3072
    featuresPerHeads = 64
elif modelName == "WinKawaks/vit-tiny-patch16-224":
    transformerHeads = 3
    transformerLayers = 12
    embeddingDim = 192
    transformerNNDim = 768
    featuresPerHeads = 64
else:
    print("model not recognised!")

model1 = ViTModel.from_pretrained(modelName)
model2 = AudioSpectogramTransformer(
    imageHeight=224, 
    imageWidth=224, 
    outputDim=0, 
    patchSize=16, 
    patchOverlap=0, 
    transformerHeads=transformerHeads, 
    embeddingDim=embeddingDim, 
    transformerLayers=transformerLayers, 
    transformerNNDim=transformerNNDim
)

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
embeddingWeight = embeddingWeight.transpose(0,2,3,1).reshape(embeddingDim,-1).T # Move the channels to the end, combine inputs.
embeddingBias = model1.embeddings.patch_embeddings.projection.bias.detach().numpy()
model2.get_layer("linear_embedding_layer").set_weights([embeddingWeight, embeddingBias])
for l in range(transformerLayers):
    transformerWeights = [
        model1.encoder.layer[l].layernorm_before.weight.detach().numpy(),
        model1.encoder.layer[l].layernorm_before.bias.detach().numpy(),
        model1.encoder.layer[l].attention.attention.query.weight.detach().numpy().T.reshape(embeddingDim, transformerHeads, featuresPerHeads),
        model1.encoder.layer[l].attention.attention.query.bias.detach().numpy().reshape(transformerHeads, featuresPerHeads),
        model1.encoder.layer[l].attention.attention.key.weight.detach().numpy().T.reshape(embeddingDim, transformerHeads, featuresPerHeads),
        model1.encoder.layer[l].attention.attention.key.bias.detach().numpy().reshape(transformerHeads, featuresPerHeads),
        model1.encoder.layer[l].attention.attention.value.weight.detach().numpy().T.reshape(embeddingDim, transformerHeads, featuresPerHeads),
        model1.encoder.layer[l].attention.attention.value.bias.detach().numpy().reshape(transformerHeads, featuresPerHeads),
        model1.encoder.layer[l].attention.output.dense.weight.detach().numpy().T.reshape(transformerHeads, featuresPerHeads, embeddingDim),
        model1.encoder.layer[l].attention.output.dense.bias.detach().numpy().reshape(embeddingDim),
        model1.encoder.layer[l].layernorm_after.weight.detach().numpy(),
        model1.encoder.layer[l].layernorm_after.bias.detach().numpy(),
        model1.encoder.layer[l].intermediate.dense.weight.detach().numpy().T,
        model1.encoder.layer[l].intermediate.dense.bias.detach().numpy(),
        model1.encoder.layer[l].output.dense.weight.detach().numpy().T,
        model1.encoder.layer[l].output.dense.bias.detach().numpy()
    ]
    name = "transformer_block_" + str(l+1)
    model2.get_layer(name).set_weights(transformerWeights)
model2.get_layer("final_normalization").set_weights([model1.layernorm.weight.detach().numpy(),model1.layernorm.bias.detach().numpy()])


# Test they give the same output
testInput = np.random.rand(1, 224, 224).astype(np.float32)
testInputTorch = torch.tensor(testInput).unsqueeze(1).repeat(1, 3, 1, 1)
with torch.no_grad():
    output1 = model1(testInputTorch).last_hidden_state
print("Model 1 output shape:", output1.shape)
print("Model 1 output",output1)
feature_model2 = tf.keras.Model(model2.input, model2.get_layer("final_normalization").output)
output2 = feature_model2(testInput)
print("Model 2 output shape:", output2.shape)
print("Model 2 output",output2)

# save model 2
feature_model2.save_weights("pre-trained_ViT_weights.h5")