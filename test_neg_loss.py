'''
pytorch中负loss是可行的
策略梯度中loss可能为负
但是不影响最终结果
'''

import torch
from torch import nn
from torch.nn.functional import softmax

x = torch.tensor([1]).float()
m = nn.Linear(in_features=1, out_features=4)
y = m(x)
print(y)

for _ in range(10):
    print('='*25 + '负loss' + '='*25)

    y = m(x)
    z = torch.tensor([1])
    criterion = nn.CrossEntropyLoss(reduction='none')
    yy = y.unsqueeze(0)
    loss = criterion(yy, z)
    neg_loss = -loss
    neg_loss.backward()

    m.weight.data -= m.weight.grad
    m.bias.data -= m.bias.grad
    m.weight.grad.zero_()
    m.bias.grad.zero_()

    print(m(x))

for _ in range(10):
    print('='*25 + '正loss' + '='*25)

    y = m(x)
    z = torch.tensor([1])
    criterion = nn.CrossEntropyLoss(reduction='none')
    yy = y.unsqueeze(0)
    loss = criterion(yy, z)
    loss.backward()

    m.weight.data -= m.weight.grad
    m.bias.data -= m.bias.grad
    m.weight.grad.zero_()
    m.bias.grad.zero_()

    print(m(x))
