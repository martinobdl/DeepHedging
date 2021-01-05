import torch


class Loss_function:

    def __init__(self):
        pass

    @classmethod
    def compute_PnL(self, y_hat, y):
        option = y
        ptf = y_hat[:, 0, -1]
        return ptf - option

    def forward(self, y_hat, y):
        raise NotImplementedError("Subclasses should implement this method")

    def __call__(self, y_hat, y):
        return self.forward(y_hat, y)


class loss_PnL(Loss_function):

    def __init__(self):
        pass

    def forward(self, y_hat, y):
        PnL = self.compute_PnL(y_hat, y)
        return torch.mean(-PnL)


class loss_MSE(Loss_function):

    def __init__(self):
        pass

    def forward(self, y_hat, y):
        PnL = self.compute_PnL(y_hat, y)
        return torch.mean(torch.pow(PnL, 2))


class loss_essInf(Loss_function):

    def __init__(self):
        pass

    def forward(self, y_hat, y):
        PnL = self.compute_PnL(y_hat, y)
        return -torch.min(PnL)


class loss_CVar(Loss_function):

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def forward(self, y_hat, y):
        PnL = self.compute_PnL(y_hat, y)
        CVar = torch.topk(-PnL, int((1-self.alpha)*PnL.shape[0])).values
        return torch.mean(CVar)

# 
# def loss_PnL(y_hat, y):
#     PnL = compute_PnL(y_hat, y)
#     return torch.mean(-PnL)
# 
# 
# def loss_MSE(y_hat, y):
#     PnL = compute_PnL(y_hat, y)
#     return torch.mean(PnL**2)
# 
# 
# def loss_essInf(y_hat, y):
#     PnL = compute_PnL(y_hat, y)
#     return -torch.min(PnL)
# 
# 
# def loss_CVar(y_hat, y, alpha=0.5):
#     PnL = compute_PnL(y_hat, y)
#     CVar = torch.topk(-PnL, int((1-alpha)*PnL.shape[0])).values
#     return torch.mean(CVar)
