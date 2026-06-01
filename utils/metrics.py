import torch

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total=0.0
        self.count=0
    def update(self,value:float,n:int):
        self.total+=value*n
        self.count+=n
    @property
    def avg(self):
        if self.count==0:
            return 0.0
        return self.total/self.count

def accuracy(logits:torch.Tensor,targets:torch.Tensor)->float:
    predictions=torch.argmax(logits,dim=1)
    correct=(predictions==targets).sum().item()
    total=targets.size(0)
    return correct/total