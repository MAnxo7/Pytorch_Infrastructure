import torch

class BasicNN(torch.nn.Module):
    def __init__(self,hidden_layer_neurons,in_features,out_features):
        super().__init__()
        self.hidden_layer_neurons = hidden_layer_neurons
        self.input = in_features
        self.output = out_features
        self.capa1 = torch.nn.Linear(self.input,self.hidden_layer_neurons)
        self.capa2 = torch.nn.Linear(self.hidden_layer_neurons,self.output)
        self.capas = [self.capa1,self.capa2]
        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)

        self.model_config = {
            "hidden_layer_neurons": self.hidden_layer_neurons,
            "in_features": self.input,
            "out_features": self.output,
        }
                   
    def forward(self,x): 
        input = self.activation(self.capas[0](x))
        # Calculate and activate all the layers until the n-1 layer
        for i in range(1,len(self.capas)-1):
            input = self.activation(self.capas[i](input))
        # Returns the last layer without activation
        return self.capas[len(self.capas)-1](input)
    
    def get_config(self):
        return self.model_config