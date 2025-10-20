import torch.nn as nn
import torch 

from tqdm import tqdm
from src.flow_matching.helpers import choose_device
from src.flow_matching.models import GenericClassifier 

def c2st(samples_1: torch.Tensor, samples_2: torch.Tensor, num_epochs=100) -> float:
    """
    Trains a binary MLP classifier to distinguish 
    between two sets of samples.

    Samples are split 80/20 into training and test set. The test set
    is used to evaluate the accuracy of the classifier.
    
    Samples should already be standardized.

    Returns:
        float: accuracy in percent
    """
    assert samples_1.shape == samples_2.shape, 'samples have different dimensions'

    # concat samples
    data = torch.cat((samples_1, samples_2), dim=0)
    num_samples, dim = data.shape

    # create targets
    targets = torch.cat((torch.ones(num_samples // 2), torch.zeros(num_samples // 2)))
    targets = targets.view(-1, 1)

    # shuffle data
    shuffled_indeces = torch.randperm(num_samples)
    data = data[shuffled_indeces]
    targets = targets[shuffled_indeces]

    # divide in training and test
    training_size = int(0.8 * num_samples)
    training_set, training_targets = data[:training_size], targets[:training_size]
    test_set, test_targets         = data[training_size:], targets[training_size:]

    # put everything on the same device
    device = choose_device()    
    classifier       = GenericClassifier(dim, hiddens=[64, 64, 32, 16], outputs=1).to(device)
    training_set     = training_set.to(device)
    training_targets = training_targets.to(device)
    test_set         = test_set.to(device)
    test_targets     = test_targets.to(device)
    
    # train the classifier 
    criterion = nn.BCELoss(reduction='sum')  
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    
    batch_size = 256
    classifier.train()

    progress_bar = tqdm(range(num_epochs))
    for epoch in progress_bar:

        batched_training_set, batched_training_targets = prepare_batches(training_set, training_targets, batch_size)
        
        total_loss = train_one_epoch(classifier, criterion, optimizer, batched_training_set, batched_training_targets)
        
        average_loss = total_loss / training_size

        progress_bar.set_description(f'epoch: {epoch} average loss: {average_loss}')

    # find accuracy 
    classifier.eval()
    outputs = classifier(test_set)
    return accuracy(outputs, test_targets)

def prepare_batches(inputs, targets, batch_size):
    """
    Shuffle and batch data. 
    """
    # shuffle data  
    shuffled_indeces = torch.randperm(inputs.shape[0])
    inputs  = inputs[shuffled_indeces]
    targets = targets[shuffled_indeces]
    
    # batches with shuffled data
    batched_inputs = torch.split(inputs, batch_size)
    batched_targets = torch.split(targets, batch_size)

    return batched_inputs, batched_targets

def train_one_epoch(model, criterion, opt, batched_inputs, batched_targets):
    
    running_loss = 0.0

    for inputs, targets in zip(batched_inputs, batched_targets):
        outputs = model(inputs)
        loss    = criterion(outputs, targets)

        opt.zero_grad()
        loss.backward()
        opt.step()

        running_loss += loss.item()

    return running_loss

def accuracy(outputs, targets):
    """
    Percentage of correct predictions.
    """
    outputs = torch.round(outputs)
    correct = torch.sum(outputs == targets).cpu().item()
    return 100 * correct / len(targets)


if __name__=="__main__":
    # dummy data
    N = 10000
    dim = 4
    samples_1 = torch.rand(size=(N, dim))
    samples_2 = torch.normal(mean=0, std=1, size=(N, dim))
    # samples_2 = torch.rand(size=(N, dim))

    accuracy = c2st(samples_1, samples_2)
    print(f'ACCURACY: {accuracy}%')