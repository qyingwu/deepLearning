from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code
    
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = CNNClassifier().to(device)
    if args.resume_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th')))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    loss = torch.nn.CrossEntropyLoss()

    train_data = load_data('data/train')
    valid_data = load_data('data/valid')

    global_step = 0
    for epoch in range(args.epoch_count):
        model.train()
        accuracy_vals = []
        loss_vals = []
        for image, label in train_data:
            image = image.to(device)
            label = label.to(device)

            logit = model(image)
            loss_val = loss(logit, label)
            accuracy_val = accuracy(logit, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
            accuracy_vals.append(accuracy_val.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        avg_acc = sum(accuracy_vals) / len(accuracy_vals)

        if train_logger:
            train_logger.add_scalar('accuracy', avg_acc, global_step)

        model.eval()

        accuracy_vals = []
        for image, label in valid_data:
            image = image.to(device)
            label = label.to(device)
            accuracy_vals.append(accuracy(model(image), label).detach().cpu().numpy())
        avg_val_acc = sum(accuracy_vals) / len(accuracy_vals)

        if valid_logger:
            valid_logger.add_scalar('accuracy', avg_val_acc, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t accuracy = %0.4f \t validate accuracy = %0.4f' % (epoch, avg_acc, avg_val_acc))
        save_model(model)

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-mo', '--momentum', type=float, default=0.9)
    parser.add_argument('-ec', '--epoch_count', type=int, default=5)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-r', '--resume_training', action='store_true')

    args = parser.parse_args()
    train(args)
