import torch

class BasicNN(torch.nn.Module):
    def __init__(self,neurons,in_features,out_features,device):
        super().__init__()
        H = neurons
        input = in_features
        output = out_features
        self.capa1 = torch.nn.Linear(input,H).to(device)
        self.capa2 = torch.nn.Linear(H,output).to(device)
        self.capas = [self.capa1,self.capa2]
        self.activation = torch.nn.LeakyReLU(negative_slope=0.1).to(device)
                   
    def forward(self,x): 
        input = self.activation(self.capas[0](x))
        # Calculate and activate all the layers until the n-1 layer
        for i in range(1,len(self.capas)-1):
            input = self.activation(self.capas[i](input))
        # Returns the last layer without activation
        return self.capas[len(self.capas)-1](input)