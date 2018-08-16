import torch


def train(model, criterion, optimizer, train_data_loader, epochs_count, experiment):
    model.train()
    for step, epoch_i in enumerate(range(epochs_count), 1):
        correct = 0
        experiment.log_current_epoch(epoch_i)
        print('Epoch: {}'.format(epoch_i))
        for sample_batched_dict in train_data_loader:
            input_batch = sample_batched_dict['input']
            label_batch = sample_batched_dict['label']
            output, loss = _training_batch_step(input_batch=input_batch, label_batch=label_batch, model=model,
                                                optimizer=optimizer, criterion=criterion)
            _, predicted = torch.max(output.data, 1)
            _, ref = torch.max(label_batch, 1)
            correct += (predicted == ref).sum()
            print('Training loss: %.3f' % loss)
            experiment.log_metric('Train loss', loss, step=step)
        accuracy = int(correct) / len(train_data_loader)
        print('Training accuracy: %.3f' % accuracy)
        experiment.log_metric('Train accuracy', accuracy, step=epoch_i)


def _training_batch_step(input_batch, label_batch, model, optimizer, criterion):
    optimizer.zero_grad()

    output = model.forward(input_batch)
    label_batch_indices = label_batch.max(1)[1]
    loss = criterion(output, label_batch_indices)
    loss.backward()
    optimizer.step()

    return output, loss

# def test(self):
#     correct = 0
#     with self.experiment.test():
#         for i, x in enumerate(self.X_test):
#             x = np.array(x)
#             x = torch.from_numpy(x).type(torch.FloatTensor).view(len(x), 1, self.config['input_size'])
#             target = torch.LongTensor(self.y_test[i])
#             outputs = self.model(x)
#             _, predicted = torch.max(outputs.data, 1)
#             correct += (predicted == torch.nonzero(target.data).squeeze(1)).sum()
#
#         accuracy = (100 * correct / len(self.y_test))
#         print('Accuracy of the network: %.3f %%' % accuracy)
#         self.experiment.log_metric('accuracy', accuracy)
