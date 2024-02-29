import torch.nn as nn
import torch
import torch.nn.functional as F

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, lora_rank=16):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.gamma_forward = nn.Parameter(torch.randn(in_features), requires_grad=False)
        self.beta = nn.Parameter(torch.randn(out_features), requires_grad=False)
        self.epsilon = 1e-8

    def binarize(self, W):
        gamma = torch.sum(torch.abs(W)) / (W.shape[0] * W.shape[1])
        W = W / (gamma + self.epsilon)
        W_bin = torch.clamp(W, -1, 1).round()
        return W + (W_bin - W).detach()


    def forward(self, input):
        if input.dim() == 1:
            input = input.unsqueeze(0)

        input_norm = F.layer_norm(input, (self.in_features,))
        quant_scale = torch.max(torch.abs(input_norm), dim=1, keepdim=True).values
        input_quant = torch.sign(input_norm) * (quant_scale / self.gamma_forward)
        
        binary_weight = self.binarize(self.weight)

        output = torch.matmul(input_quant, binary_weight.t())
        output = output * self.beta.expand_as(output)

        return output