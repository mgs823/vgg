import torch
input=torch.arange(0,6)
print(input)
print(input.shape)
input1=input.unsqueeze(0)
print(input1.unsqueeze(0))
print(input1.unsqueeze(0).shape)
print(input1.unsqueeze(1))
print(input1.unsqueeze(1).shape)
print(input1.unsqueeze(2))
print(input1.unsqueeze(2).shape)




