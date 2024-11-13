import torch
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):
    def __init__(self, layers=[16, 32, 64, 128], n_in=3, n_out=6, kernel_size=3):
        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        # raise NotImplementedError('CNNClassifier.__init__')
        super().__init__()
        self.input_mean = torch.Tensor([0.3233, 0.3301, 0.3441])
        self.input_std = torch.Tensor([0.2531, 0.2226, 0.2490])

        L = []
        c = n_in
        for l in layers:
            L.append(self.Block(c, l, kernel_size, 2))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_out)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        # raise NotImplementedError('CNNClassifier.forward')
        z = self.network((x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device))
        return self.classifier(z.mean(dim=[2, 3]))

    class Block(torch.nn.Module):
        def __init__(self, n_in, n_out, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_in, n_out, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride, bias=False)
            self.c2 = torch.nn.Conv2d(n_out, n_out, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.c3 = torch.nn.Conv2d(n_out, n_out, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.b1 = torch.nn.BatchNorm2d(n_out)
            self.b2 = torch.nn.BatchNorm2d(n_out)
            self.b3 = torch.nn.BatchNorm2d(n_out)
            self.skip = torch.nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))


class FCN(torch.nn.Module):
    def __init__(self, layers=[16, 32, 64, 128], n_in=3, n_out=5, kernel_size=3, use_skip=True):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        # raise NotImplementedError('FCN.__init__')
        self.input_mean = torch.Tensor([0.2788, 0.2657, 0.2629])
        self.input_std = torch.Tensor([0.2064, 0.1944, 0.2252])

        c = n_in
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[0:3]
        i = 0
        for l in layers:
            self.add_module('conv%d' % i, CNNClassifier.Block(c, l, kernel_size, 2))
            c = l
            i += 1
        for l in reversed(layers):
            i -= 1
            self.add_module('upconv%d' % i, self.Ublock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(c, n_out, 1)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        # raise NotImplementedError('FCN.forward')
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
        up_act = []
        for i in range(self.n_conv):
            up_act.append(z)
            z = self._modules['conv%d'%i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d'%i](z)
            z = z[:, :, :up_act[i].size(2), :up_act[i].size(3)]
            if self.use_skip:
                z = torch.cat([z, up_act[i]], dim=1)
        return self.classifier(z)

    class Ublock(torch.nn.Module):
        def __init__(self, n_in, n_out, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
