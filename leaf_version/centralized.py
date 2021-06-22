import process_data
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.dataset import ArrayDataset
import time


# Data
train_data = process_data.read_all_data_in_dir("../data/femnist/train")
valid_data = process_data.read_all_data_in_dir("../data/femnist/test")

# Convert to mxnet ndarray
X_train = nd.array(train_data[0])
y_train = nd.array(train_data[1])
X_valid = nd.array(valid_data[0])
y_valid = nd.array(valid_data[1])

# Reshape 784 array to (1, 28, 28) matrix (channel, height, width)
X_train = nd.reshape(X_train, shape=(0, 1, 28, 28))
X, y = X_train[0], y_train[0]
print('X shape: ', X.shape, 'X dtype', X.dtype, 'y:', y)
X_valid = nd.reshape(X_valid, shape=(0, 1, 28, 28))

# Pass data and labels into ArrayDataset
dataset = ArrayDataset(X_train, y_train)
dataset_valid = ArrayDataset(X_valid, y_valid)

# Transform normalize ???

# Use mxnet dataloader to batch data
batch_size = 100
training_data = gluon.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, last_batch='discard')
valid_data = gluon.data.DataLoader(
    dataset_valid, batch_size=batch_size, shuffle=True, last_batch='discard')

# for data, label in training_data:
#     print(data.shape, label.shape)
#     break

# Model
net = nn.Sequential()
net.add(nn.Conv2D(channels=32, kernel_size=[5, 5], activation='relu'),
        nn.MaxPool2D(pool_size=[2, 2], strides=2),
        nn.Conv2D(channels=64, kernel_size=[5, 5], activation='relu'),
        nn.MaxPool2D(pool_size=[2, 2], strides=2),
        nn.Flatten(),
        nn.Dense(2048, activation="relu"),
        nn.Dense(62))
net.initialize(init=init.Xavier())

# Loss
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# Trainer: sgd, learning rate
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# Accuracy
def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (output.argmax(axis=1) ==
            label.astype('float32')).mean().asscalar()

# Training loop
for epoch in range(1, 101):
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time.time()
    for data, label in training_data:
        # forward + backward
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        # update parameters
        trainer.step(batch_size)
        # calculate training metrics
        train_loss += loss.mean().asscalar()
        train_acc += acc(output, label)
    # calculate validation accuracy
    for data, label in valid_data:
        valid_acc += acc(net(data), label)
    print("Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
            epoch, train_loss/len(training_data), train_acc/len(training_data),
            valid_acc/len(valid_data), time.time()-tic))

# Save model
# net.save_parameters('net.params')