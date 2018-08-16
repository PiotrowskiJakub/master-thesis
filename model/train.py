import torch


def train(model, criterion, optimizer, train_data_loader, dev_data_loader, epochs_count, experiment):
    for step, epoch_i in enumerate(range(epochs_count), 1):
        model.train()
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
        _evaluate_model_after_epoch(model=model, data_loader=dev_data_loader, epoch_i=epoch_i, experiment=experiment)


def _training_batch_step(input_batch, label_batch, model, optimizer, criterion):
    optimizer.zero_grad()

    output = model.forward(input_batch)
    label_batch_indices = label_batch.max(1)[1]
    loss = criterion(output, label_batch_indices)
    loss.backward()
    optimizer.step()

    return output, loss


def _evaluate_model_after_epoch(model, data_loader, epoch_i, experiment):
    model.eval()
    with torch.no_grad():
        correct = 0
        for sample_batched_dict in data_loader:
            input_batch = sample_batched_dict['input']
            label_batch = sample_batched_dict['label']
            output = model.forward(input_batch)
            _, predicted = torch.max(output.data, 1)
            _, ref = torch.max(label_batch, 1)
            correct += (predicted == ref).sum()
        accuracy = int(correct) / len(data_loader)
        print('Dev accuracy: %.3f' % accuracy)
        experiment.log_metric('Dev accuracy', accuracy, step=epoch_i)
