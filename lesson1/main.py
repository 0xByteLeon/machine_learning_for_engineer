import matplotlib.pyplot as plt

weight = 1
bias = 0

learning_rate = 0.01

training_set = [(2, 5), (5, 11), (6, 13), (7, 15), (8, 17)]
validating_set = [(12, 25), (1, 3)]
testing_set = [(9, 19), (13, 27)]

# Lists to store epochs, weights, and biases for plotting
epochs = []
weights = []
biases = []

for epoch in range(1, 10000):
    print(f'epoch: {epoch}')
    epochs.append(epoch)
    for x, y in training_set:
        # predict y
        predicted = weight * x + bias
        # calculate loss
        loss = predicted - y

        print(f'training epoch: {epoch}, x: {x}, y: {y}, predicted: {predicted}, loss: {loss}, weight: {weight}, bias: {bias}')
        if loss < 0:
            weight = weight + learning_rate * x
            bias = bias + learning_rate
        else:
            weight = weight - learning_rate * x
            bias = bias - learning_rate

        weights.append(weight)
        biases.append(bias)

    validating_accuracy = 0
    for x, y in validating_set:
        predicted = weight * x + bias
        validating_accuracy += 1 - abs(predicted - y) / y
    validating_accuracy /= len(validating_set)

    print(f'validating epoch: {epoch}, accuracy: {validating_accuracy}')
    if validating_accuracy > 0.99:
        break

testing_accuracy = 0
for x, y in testing_set:
    predicted = weight * x + bias
    testing_accuracy += 1 - abs(predicted - y) / y
testing_accuracy /= len(testing_set)
print(f'testing accuracy: {testing_accuracy}')

# plt.figure(figsize=(10,6))
plt.plot(weights, label='weight')
plt.plot(biases, label='bias')
plt.legend()
plt.show()