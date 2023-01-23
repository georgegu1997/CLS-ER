import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearDecomposed(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.dim = dim

        # Projection is A = U S V^T
        self.U = nn.Parameter(torch.eye(dim, dim), requires_grad=False)
        self.S = nn.Parameter(torch.ones(dim))
        self.V = nn.Parameter(torch.eye(dim, dim))
        self.feat_mean = nn.Parameter(torch.zeros(1, dim), requires_grad=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter('bias', None)

        self.V_init = None

    def forward(self, x):
        '''
        x: (N, dim)
        '''
        x = x - self.feat_mean
        x_hat = F.linear(x, self.U @ torch.diag(self.S) @ self.V.t(), self.bias)
        x_hat = x_hat + self.feat_mean

        return x_hat

    def set_parameters(self, U, feat_mean, init_identity) -> None:
        if self.V_init is not None:
            print((self.V_init - self.V).norm())
        
        with torch.no_grad():
            self.U.copy_(U)
            self.S.fill_(1.0)
            
            if init_identity:
                self.V.copy_(U)
                self.bias.fill_(0.0)
            else:
                # Random init
                nn.init.kaiming_normal_(self.V, a = math.sqrt(5))
                if self.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.V)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(self.bias, -bound, bound)

            self.feat_mean.copy_(feat_mean)
            self.V_init = self.V.detach().clone()

    def soft_reg_loss(self, w):
        loss_S = (w * (self.S - 1.0)).norm(2)
        loss_V = (self.V.T @ self.V - torch.eye(self.dim, device=self.V.device)).norm(2)
        return loss_S, loss_V
