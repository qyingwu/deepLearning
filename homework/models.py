import torch


class CNNClassifier(torch.nn.Module):
    def __init__(self, layers=[16, 32, 64, 128], n_in=3, n_out=6, kernel_size=5):
        """
        Your code here
        """
        # raise NotImplementedError('CNNClassifier.__init__')
        super().__init__()

        L = []
        c = n_in
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, kernel_size, stride=2, padding=kernel_size//2))
            L.append(torch.nn.ReLU())
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_out)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        # raise NotImplementedError('CNNClassifier.forward')
        return self.classifier(self.network(x).mean(dim=[2, 3]))


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r