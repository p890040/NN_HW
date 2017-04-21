import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x_data = np.loadtxt('in_train.txt')/100
y_data = np.loadtxt('out_train.txt')[: , np.newaxis]

W1 = -1 + 2*np.random.rand(100,16)
W2 = -1 + 2*np.random.rand(50,100)
W3 = -1 + 2*np.random.rand(25,50)
W4 = -1 + 2*np.random.rand(20,25)
W5 = -1 + 2*np.random.rand(10,20)
B1 = -1 + 2*np.random.rand(100,1)
B2 = -1 + 2*np.random.rand(50,1)
B3 = -1 + 2*np.random.rand(25,1)
B4 = -1 + 2*np.random.rand(20,1)
B5 = -1 + 2*np.random.rand(10,1)

learning_rate = 0.25
epho = 20
date_length = len(x_data)

for e in range(epho):
    count = 0
    for i in range(date_length):
        a0 = x_data[i].reshape(16,1)    #一維矩陣用transpose()轉置會有問題，故用此方法
        n1 = np.dot(W1, a0) + B1        #numpy array中(跟numpy matix不同)，dot相當於一般矩陣相乘(broadcasting)，*是element-wise的乘法
        a1 = sigmoid(n1)
        n2 = np.dot(W2, a1) + B2
        a2 = sigmoid(n2)
        n3 = np.dot(W3, a2) + B3
        a3 = sigmoid(n3)
        n4 = np.dot(W4, a3) + B4
        a4 = sigmoid(n4)
        n5 = np.dot(W5, a4) + B5;
        a5 = softmax(n5)

        t = np.zeros((10,1))
        t[(int)(y_data[i])] = 1

        # np.diag()接收一維矩陣時才會是擴增效果，接收二維以上會找對角線return一維矩陣，故reshape成一維
        # #(10,1)=>2維 (10,)=>1維
        F_5 = np.diag(np.dot(np.diag(a5.reshape(len(a5))), (1 - a5)).reshape(len(a5)))
        error5 = -2 * np.dot(F_5, (t - a5))
        F_4 = np.diag(np.dot(np.diag(a4.reshape(len(a4))), (1 - a4)).reshape(len(a4)))
        error4 = np.dot(F_4, (np.dot(W5.transpose(), error5)))
        F_3 = np.diag(np.dot(np.diag(a3.reshape(len(a3))), (1 - a3)).reshape(len(a3)))
        error3 = np.dot(F_3, (np.dot(W4.transpose(), error4)))
        F_2 = np.diag(np.dot(np.diag(a2.reshape(len(a2))), (1 - a2)).reshape(len(a2)))
        error2 = np.dot(F_2, (np.dot(W3.transpose(), error3)))
        F_1 = np.diag(np.dot(np.diag(a1.reshape(len(a1))), (1 - a1)).reshape(len(a1)))
        error1 = np.dot(F_1, (np.dot(W2.transpose(), error2)))

        delta_w5 = np.dot(error5, a4.transpose())
        delta_w4 = np.dot(error4, a3.transpose())
        delta_w3 = np.dot(error3, a2.transpose())
        delta_w2 = np.dot(error2, a1.transpose())
        delta_w1 = np.dot(error1, a0.transpose())
        delta_b5 = error5
        delta_b4 = error4
        delta_b3 = error3
        delta_b2 = error2
        delta_b1 = error1

        W5 = W5 - learning_rate * delta_w5
        W4 = W4 - learning_rate * delta_w4
        W3 = W3 - learning_rate * delta_w3
        W2 = W2 - learning_rate * delta_w2
        W1 = W1 - learning_rate * delta_w1
        B5 = B5 - learning_rate * delta_b5
        B4 = B4 - learning_rate * delta_b4
        B3 = B3 - learning_rate * delta_b3
        B2 = B2 - learning_rate * delta_b2
        B1 = B1 - learning_rate * delta_b1

        if (np.argmax(a5) == y_data[i]):
            count += 1
    print('epho ',e+1,' Train Correct Rate : ', count / date_length)

#存儲變數，W1=W1，前者是之後讀取要用的名稱
np.savez("train_weight.npz", W1=W1, W2=W2, W3=W3, W4=W4, W5=W5, B1=B1, B2=B2, B3=B3, B4=B4, B5=B5)
