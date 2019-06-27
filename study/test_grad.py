"""Perform vowel recognition training.
"""

import torch
import wavetorch
import numpy as np
from torch.nn.functional import conv2d


def _laplacian(y, h):
    """Laplacian operator"""
    operator = h**(-2) * torch.tensor([[[[0.0,  1.0, 0.0], [1.0, -4.0, 1.0], [0.0,  1.0, 0.0]]]])
    return conv2d(y.unsqueeze(1), operator, padding=1).squeeze(1)

def step(b, c, y1, y2, dt, h):
        y = torch.mul((dt.pow(-2) + b * dt.pow(-1)).pow(-1),
              (2/dt.pow(2)*y1 - torch.mul( (dt.pow(-2) - b * dt.pow(-1)), y2)
                       + torch.mul(c.pow(2), _laplacian(y1, h)))
             )
        return y

class HardcodedStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, c, y1, y2, dt, h):
        ctx.save_for_backward(b, c, y1, y2, dt, h)
        return step(b, c, y1, y2, dt, h)

    @staticmethod
    def backward(ctx, grad_output):
        b, c, y1, y2, dt, h = ctx.saved_tensors

        grad_b = grad_c = grad_y1 = grad_y2 = grad_dt = grad_h = None

        if ctx.needs_input_grad[0]:
            grad_b = - (dt * b + 1).pow(-2) * dt * (c.pow(2) * dt.pow(2) * _laplacian(y1, h) + 2*y1 - 2 * y2 ) * grad_output
        if ctx.needs_input_grad[1]:
            grad_c = (b*dt + 1).pow(-1) * (2 * c * dt.pow(2) * _laplacian(y1, h) ) * grad_output
        if ctx.needs_input_grad[2]:
            grad_y1 = (c.pow(2) * dt.pow(2) * _laplacian(grad_output, h) + 2*grad_output) * (b*dt + 1).pow(-1)
        if ctx.needs_input_grad[3]:
            grad_y2 = (b*dt -1) * (b*dt + 1).pow(-1) * grad_output

        return grad_b, grad_c, grad_y1, grad_y2, grad_dt, grad_h

N = 5
M = 1
params = [
    torch.rand(N,N, requires_grad=True), # b
    torch.rand(N,N, requires_grad=True), # c
    torch.rand(M, N,N, requires_grad=True), # y1
    torch.rand(M, N,N, requires_grad=True), # y2
    torch.tensor(0.10),
    torch.tensor(0.25)
]

# Take derivative using pytorch's AD
y_ad = step(*params)
y_ad.sum().backward()

params_grad_ad = []
for param in params:
    if param.grad is not None:
        params_grad_ad.append(param.grad.clone())
        param.grad.detach_()
        param.grad.zero_() # Detach and zero for subsequent calculations

y_hardcoded = HardcodedStep.apply(*params)
y_hardcoded.sum().backward() # Do the backprop

params_grad_hard = []
for param in params:
    if param.grad is not None:
        params_grad_hard.append(param.grad.clone())
        param.grad.detach_()
        param.grad.zero_() # Detach and zero for subsequent calculations

# Compare
diff = [(params_grad_hard[i]-params_grad_ad[i]).norm().item() for i in range(0,len(params_grad_hard))]
print(diff)
