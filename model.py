import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size, 
                            hidden_size=self.hidden_size,
                            num_layers=num_layers,
                            batch_first = True)

        self.fc = nn.Linear(hidden_size, vocab_size)
            
    def forward(self, features, captions):
        
        captions = captions[:,:-1]
        
        embeddings = self.embed(captions)
        
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        x, _ = self.lstm(embeddings)
        
        x = self.fc(x)

        return x

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        output_length = 0
        
        
        while (output_length <= max_len):
        
            output, states = self.lstm(inputs, states)

            output = self.fc(output)
#             _, predicted_index = output.max(2)
            predicted_index = torch.argmax(output, 2)

            outputs.append(predicted_index.item())


            if (predicted_index == 1):
                break

            inputs = self.embed(predicted_index)

            output_length +=1
        
        
        return outputs
    
    
    
    
    
    
    
    