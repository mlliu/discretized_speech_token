from transformers import AutoProcessor, HubertModel
import torch
from datasets import load_dataset
import faiss

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

# load the model
processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

# get the 73 array data from the dataset
array_data = []
for i in range(len(dataset)):
    array_data.append(dataset[i]["audio"]["array"])

inputs = processor(array_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, output_attentions=True)


last_hidden_states = outputs.last_hidden_state # the real-value representation of the input audio

#extract_features = outputs.extract_features # this is the output of the last conv layer of the model
print(last_hidden_states.shape) # (batch_size, sequence_length, hidden_size) torch.Size([73, 292, 768])


# reshape the last_hidden_states to shape (batch_size*sequence_length, hidden_size)
x = last_hidden_states.reshape(-1, last_hidden_states.shape[-1]) # (batch_size*sequence_length, hidden_size)

print("shape of extract feature from 73 examples: ",x.shape) # torch.Size([21256, 768])

# using k-means clustering to quantize the real-value representation of the input audio
ncentroids = 100 #1024
niter = 20
verbose = True
d = x.shape[1]
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
kmeans.train(x)
print("shape of kmeans centroids: ", kmeans.centroids.shape) # (100, 768)



