import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

r = np.load("train_weight.npz")
W1 = r["W1"]
W2 = r["W2"]
W3 = r["W3"]
W4 = r["W4"]
W5 = r["W5"]
B1 = r["B1"]
B2 = r["B2"]
B3 = r["B3"]
B4 = r["B4"]
B5 = r["B5"]
x_data_test = np.loadtxt('in_test.txt')/100
y_data_test = np.loadtxt('out_test.txt')[: , np.newaxis]
count = 0
date_length = len(x_data_test)

for i in range(date_length):
    a0 = x_data_test[i].reshape(16, 1)
    n1 = np.dot(W1, a0) + B1
    a1 = sigmoid(n1)
    n2 = np.dot(W2, a1) + B2
    a2 = sigmoid(n2)
    n3 = np.dot(W3, a2) + B3
    a3 = sigmoid(n3)
    n4 = np.dot(W4, a3) + B4
    a4 = sigmoid(n4)
    n5 = np.dot(W5, a4) + B5;
    a5 = sigmoid(n5)

    if(np.argmax(a5) == y_data_test[i]):
        count+=1

print('Correct Rate : ',count/date_length)