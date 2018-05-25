import torch
import torch.nn.functional as F


def pad(tensors):
    max_length = max(tensors, key=lambda tensor: tensor.size(0)).size(0)
    padded_tensors = []
    for tensor in tensors:
        padding_length = max_length - tensor.size(0)
        if padding_length == 0:
            padded_tensors.append(tensor)
            continue
        features = []
        for dim in range(tensor.size(1)):
            feature = tensor[:, dim]
            padded_value = feature[-1]
            features.append(F.pad(feature, (0, padding_length), value=padded_value))
        padded_tensors.append(torch.stack(features, dim=1))

    return torch.stack(padded_tensors)


def pad_batch(samples):
    input_tensors = list(map(lambda sample: sample['input'], samples))
    label_tensors = list(map(lambda sample: sample['label'], samples))
    padded_input_tensors = pad(input_tensors)

    return {
        'input': padded_input_tensors,
        'label': torch.stack(label_tensors)
    }
