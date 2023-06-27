import json
import torch
from networker import FC
import torch.nn.functional as torch_fun
from tools import *


if __name__ == '__main__':
    ############################
    #   Parameter
    ############################
    train_room_id = [1, 3, 4, 5, 7, 9, 10, 11, 13]  # Room number used for training
    test_room_id = [2, 6, 8, 12, 14, 15]            # Room number used for testing

    # Task mode:
    # 4 indicates whether the classification is open or closed
    # 5 indicates whether the classification is occupied or unoccupied
    task_mode = 4
    learning_rate = 0.00005         # Learning rate
    epochs = 20000                  # Epochs max
    train_batch_size = 1024         # Batch size for train
    test_batch_size = -1            # Batch size for test
    always_draw = True              # Whether it is always drawn, set to true will affect the speed


    ############################
    #   1. Data Set Loading
    ############################
    print('[1]. Data Set Loading...')

    fxs = np.load('data/WNROOM/X.npy')     # Frequency domain data after Fourier transform
    ys = np.load('data/WNROOM/Y.npy')      # Labels for X

    ############################
    #   2. Dataset Divided
    ############################
    print('[2]. Data will be divided by room number: train:{}\ttest:{}'.format(train_room_id, test_room_id))

    rid_indexes = build_index_data_by_room(ys[:, [1, task_mode]])   # Get an indexed dict with the room number as the key

    train_indexes = np.concatenate([rid_indexes[room_id] for room_id in train_room_id])     # Get train set indexes
    test_indexes = np.concatenate([rid_indexes[room_id] for room_id in test_room_id])       # Get test set indexes
    if test_batch_size < 0:
        test_batch_size = len(test_indexes)

    print('\tsize of Train set: {}\t size of Test set: {}'.format(len(train_indexes), len(test_indexes)))

    ###############################################
    #   3. Define neural networks and optimizers
    ##############################################
    print('[3]. A fully connected network is used for training(Cross-room), Using sgd optimization, Batch_size:{}'.
          format(len(train_room_id), train_batch_size))
    net = FC(2)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    ############################
    #   4. Train and Test
    ############################
    epoch_loss = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    if torch.cuda.is_available():
        net.cuda()
    loss_plot = LossPlot()
    for epoch in range(epochs):
        # Train
        net.train()
        np.random.shuffle(train_indexes)
        train_batch_loss = {
            'train_loss': [],
            'train_acc': []
        }
        for i in range(len(train_indexes) // train_batch_size):
            # Get data of a batch_size
            batch_indexes = train_indexes[i * train_batch_size:(i + 1) * train_batch_size]
            batch_x = fxs[batch_indexes, :]
            batch_y = ys[batch_indexes, task_mode]

            # Preprocess the datas
            batch_x = torch.tensor(batch_x).float()
            batch_y = torch.tensor(batch_y).long()
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

            # Train
            optimizer.zero_grad()
            outputs = net(batch_x)                              # Forward pass
            loss = torch_fun.cross_entropy(outputs, batch_y)    # Get loss
            loss.backward()                                     # backward pass
            optimizer.step()                                    # Gradient descent

            # Collect the data
            train_batch_loss['train_loss'].append(loss.item())
            train_batch_loss['train_acc'].append(acc_fun(outputs.detach().cpu().numpy(), batch_y.cpu().numpy()))

        # Test
        test_batch_loss = {
            'test_loss': [],
            'test_acc': []
        }
        net.eval()
        for i in range(len(test_indexes) // test_batch_size):
            batch_indexes = test_indexes[i * train_batch_size:(i + 1) * train_batch_size]
            batch_x = fxs[batch_indexes, :]
            batch_y = ys[batch_indexes, task_mode]

            batch_x = torch.tensor(batch_x).float()
            batch_y = torch.tensor(batch_y).long()
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

            outputs = net(batch_x)
            label_loss = torch_fun.cross_entropy(outputs, batch_y)
            acc = acc_fun(outputs.detach().cpu().numpy(), batch_y.cpu().numpy())
            test_batch_loss['test_loss'].append(label_loss.item())
            test_batch_loss['test_acc'].append(acc)

        # Collect the data
        epoch_loss['train_loss'].append(np.average(train_batch_loss['train_loss']))
        epoch_loss['test_loss'].append(np.average(test_batch_loss['test_loss']))
        epoch_loss['train_acc'].append(np.average(train_batch_loss['train_acc']))
        epoch_loss['test_acc'].append(np.average(test_batch_loss['test_acc']))

        # Print
        print('[Epoch:{}]\t train_loss:{:.5f} \t train_acc:{:.5f} \t test_loss:{:.5f} \t test_acc:{:.5f}'.format(
                epoch, epoch_loss['train_loss'][-1], epoch_loss['train_acc'][-1],
                epoch_loss['test_loss'][-1], epoch_loss['test_acc'][-1]
            ))

        if epoch % 20 == 0 or epoch < 100 or always_draw:
            loss_plot.plot(epoch_loss, ['train_acc', 'test_acc'])

        if epoch % 100 == 99:
            with open('eg.json', 'w') as f:
                json.dump(epoch_loss, f)