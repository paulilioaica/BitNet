import torch.nn as nn
import torch
import torch.nn.functional as F

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, a=-1, b=1, epsilon = 1e-5):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.epsilon = epsilon
        self.a = a
        self.b = b
        self.in_features = in_features
        self.out_features = out_features
        self.gamma_forward = nn.Parameter(torch.ones(in_features))
        self.beta_forward = nn.Parameter(torch.ones(out_features))
    
    def get_binary_weight(self):
        Wb = self.binarize(self.weight)
        return Wb
        
    def round_clip(self, W, a=-1, b=1):
        # make sure we broadcast a and b to the same shape as W
        a = a * torch.ones_like(W)
        b = b * torch.ones_like(W)
        W = torch.max(a, torch.min(b, W.round()))
        return W

    def binarize(self, W):  
        gamma = torch.sum(torch.abs(W)) / (W.shape[0] * W.shape[1])  
        W = W / (gamma + self.epsilon)  
        W_bin = self.round_clip(W, self.a, self.b)  
        W = W + (W_bin - W).detach()  # STE for the rounding operation  
        return torch.nn.Parameter(W, requires_grad=True)  
    
def forward(self, input):
    # Ensure input is at least 2D
    if input.dim() == 1:
        input = input.unsqueeze(1)

    input_norm = F.layer_norm(input, (self.in_features,))

    # Absmax Quantization
    quant_scale = torch.max(torch.abs(input_norm), dim=1, keepdim=True).values
    input_quant = torch.sign(input_norm) * (quant_scale / self.gamma_forward)
    
    weights_bin = self.get_binary_weight()

    # Calculate the positive and negative parts of the weight
    weight_pos = torch.clamp(weights_bin, min=0)
    weight_neg = torch.clamp(weights_bin, max=0)

    # Calculate the output as the sum of the positive and negative parts
    output_pos = torch.sum(input_quant * weight_pos, dim=1)
    output_neg = torch.sum(input_quant * weight_neg, dim=1)
    output = output_pos + output_neg


    return output

