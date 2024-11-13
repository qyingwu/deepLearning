from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch

def train(args):
    model = model_factory[args.model]()

    """
    Your code here

    """
    # raise NotImplementedError('train')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    loss = ClassificationLoss()
  
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    trainData = load_data('data/train')
    validData = load_data('data/valid')

    for epoch in range(args.epoch_count):
        model.train()
        lossVals = []
        accuracyVals = []

        for image, label in trainData:
            image = image.to(device)
            label = label.to(device)

            logit = model(image)
            lossVal = loss(logit, label)
            accuracyVal = accuracy(logit, label)

            lossVals.append(lossVal.detach().cpu().numpy())
            accuracyVals.append(accuracyVal.detach().cpu().numpy())

            optimizer.zero_grad()
            lossVal.backward()
            optimizer.step()

        model.eval()

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-mo', '--momentum', type=float, default=0.95)
    parser.add_argument('-ec', '--epoch_count', type=int, default=60)

    args = parser.parse_args()
    train(args)
