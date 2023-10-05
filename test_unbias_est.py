
from collections import Counter
import torch

x = torch.tensor([0.1, 0.4, 0.3, 0.2])

r = []
times = 100000
for _ in range(times):
    tmp = torch.multinomial(x, num_samples=1)[0]
    r.append(tmp.item())

rr = Counter(r)
for key, value in rr.items():
    rr[key] = value / times

for i in range(len(x)):
    print("{}: {}".format(i, rr[i]))
