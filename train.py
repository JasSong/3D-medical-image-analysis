import numpy as np
import tensorflow as tf
import os
import pickle
from train_test import mk_train_test
# from sklearn.model_selection import train_test_split

# DIR_0 = "C://Users//jpmyo//Desktop//jason//feature_30//0"
# DIR_1 = "C://Users//jpmyo//Desktop//jason//feature_30//1"
# train_pkl_path_0 = "C://Users//jpmyo//Desktop//jason//model1//Train_0"
# train_pkl_path_1 = "C://Users//jpmyo//Desktop//jason//model1//Train_1"
# test_pkl_path_0 = "C://Users//jpmyo//Desktop//jason//model1//Test_0"
# test_pkl_path_1 = "C://Users//jpmyo//Desktop//jason//model1//Test_1"

# train_pkl_path_0 = "C://Users//korea//Desktop//project//sungmo//model1//Train_0"
# train_pkl_path_1 = "C://Users//korea//Desktop//project//sungmo//model1//Train_1"
# test_pkl_path_0 = "C://Users//korea//Desktop//project//sungmo//model1//Test_0"
# test_pkl_path_1 = "C://Users//korea//Desktop//project//sungmo//model1//Test_1"

train_pkl_path_0 = "C://Users//korea//Desktop//project//sungmo//feature_30//0"
train_pkl_path_1 = "C://Users//korea//Desktop//project//sungmo//feature_30//1"
test_pkl_path_0 = "C://Users//korea//Desktop//project//sungmo//feature_30//0"
test_pkl_path_1 = "C://Users//korea//Desktop//project//sungmo//feature_30//1"

#Hyperparameter
n_inputs = 1024 * 30  # input data
learning_rate = 0.001  # learningrate
momentum = 0.008  # momentum
n_epochs = 500  # epoch 20
batch_size = 10  # batch size
dropout_rate = 0.8  # dropout rate

# hidden layers
n_hidden0 = 300
n_hidden1 = 300  # first hiddenlayer nodes
n_hidden2 = 300  # second hiddenlayer nodes
n_hidden3 = 300
n_hidden4 = 300
n_hidden5 = 300
# ouput layers
n_outputs = 2  # classification 2

# seed = [30, 23425, 1302, 23, 412, 45768, 102934, 5467, 287234, 12383, 123040, 12314, 1235674, 34654356, 635465436354, 718429, 142697, 1289042]
seed = [30, 23425, 1302, 23, 412, 45768, 102934, 5467, 287234, 12383]

def reset_graph():
    tf.reset_default_graph()

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

training = tf.placeholder_with_default(False, shape=(), name='training')
X_drop = tf.layers.dropout(X, dropout_rate, training=training)
with tf.name_scope("dnn"):
    hidden0 = tf.layers.dense(X_drop, n_hidden0, activation=tf.nn.relu,
                              name="hidden0")  # use default initializer in dense method
    hidden0_drop = tf.layers.dropout(hidden0, dropout_rate,
                                     training=training)  # use default initializer in dense method
    hidden1 = tf.layers.dense(hidden0_drop, n_hidden1, activation=tf.nn.relu,
                              name="hidden1")  # use default initializer in dense method
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate,
                                     training=training)  # use default initializer in dense method
    hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, activation=tf.nn.relu,
                              name="hidden2")  # use default initializer in dense method
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate,
                                     training=training)  # use default initializer in dense method
    hidden3 = tf.layers.dense(hidden2_drop, n_hidden3, activation=tf.nn.relu,
                              name="hidden3")  # use default initializer in dense method
    hidden3_drop = tf.layers.dropout(hidden3, dropout_rate,
                                     training=training)  # use default initializer in dense method
    hidden4 = tf.layers.dense(hidden3_drop, n_hidden4, activation=tf.nn.relu,
                              name="hidden4")  # use default initializer in dense method
    hidden4_drop = tf.layers.dropout(hidden4, dropout_rate,
                                     training=training)  # use default initializer in dense method
    hidden5 = tf.layers.dense(hidden4_drop, n_hidden5, activation=tf.nn.relu,
                              name="hidden5")  # use default initializer in dense method
    hidden5_drop = tf.layers.dropout(hidden5, dropout_rate,
                                     training=training)  # use default initializer in dense method

    logits = tf.layers.dense(hidden5_drop, n_outputs, name="outputs")  # use default initializer in dense method

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)  # use sparse softmax cross entropy function (classification)
    loss = tf.reduce_mean(xentropy, name="loss")
    tf.summary.scalar('loss', loss)
with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum,
                                           use_nesterov=True)  # Accelrated nesterove Momentum optimizer
    training_op = optimizer.minimize(loss)
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

train_y_0 = np.zeros(shape = 134)
train_y_1 = np.ones(shape = 134)
test_y_0 = np.zeros(shape = 124)
test_y_1 = np.ones(shape = 34)

total_acc = []
for seed_num in seed :
    train0_path_list, train1_path_list , test0_path_list, test1_path_list = mk_train_test(134,seed_num)

#########
    # Train
    train0_arr = np.zeros(shape=(134, 1024 * 30))
    # train0_path_list = os.listdir(train_pkl_path_0) #
    train0_path_list = list(train0_path_list)
    for X0_con in range(len(train0_path_list)):
        with open(os.path.join(train_pkl_path_0, train0_path_list[X0_con]), 'rb') as f:
            train0_arr[X0_con] = pickle.load(f)

    train1_arr = np.zeros(shape=(134, 1024 * 30))
    # train1_path_list = os.listdir(train_pkl_path_1) #
    train1_path_list = list(train1_path_list)
    for X1_con in range(len(train0_path_list)):
        with open(os.path.join(train_pkl_path_1, train1_path_list[X1_con]), 'rb') as f:
            train1_arr[X1_con] = pickle.load(f)

    # Test
    test_arr = np.zeros(shape=(124 + 34, 1024 * 30))
    test_y = np.append(test_y_0, test_y_1)
    # test0_path_list = os.listdir(test_pkl_path_0)  #
    # test1_path_list = os.listdir(test_pkl_path_1)  #
    test_total_path = list(test0_path_list) + list(test1_path_list)
    for test in range(len(test0_path_list)):
        with open(os.path.join(test_pkl_path_0, test_total_path[test]), 'rb') as f:
            test_arr[test] = pickle.load(f)
    for test in range(len(test0_path_list), (len(test0_path_list) + len(test1_path_list))):
        with open(os.path.join(test_pkl_path_1, test_total_path[test]), 'rb') as f:
            test_arr[test] = pickle.load(f)

    print(train0_arr.shape)
    print(train0_arr.shape)

    init = tf.global_variables_initializer()  # initialize parameter variables
    saver = tf.train.Saver()


    with tf.Session() as sess:
        init.run()
        accuracy_test = []
        train_writer = tf.summary.FileWriter('./logs' +'/train', sess.graph)  # 저장디렉터리 설정
        test_writer = tf.summary.FileWriter('./logs' + '/test')
        save_path = saver.save(sess, "./model1_final.ckpt")
        for epoch in range(n_epochs):
            i = 0
            for batch in range((len(train0_path_list) // batch_size)):
                X_batch = np.append(train0_arr[i: i + batch_size], train1_arr[i:i + batch_size])
                # print(X_batch.reshape(20,-1).shape)
                y_batch = np.append(train_y_0[i: i + batch_size], train_y_1[i: i + batch_size])
                # print(y_batch)
                i += batch_size
                sess.run(training_op, feed_dict={X: X_batch.reshape(20, -1), y: y_batch.reshape(-1)})
            acc_train_0 = accuracy.eval(feed_dict={X: train0_arr, y: train_y_0.reshape(-1)})
            acc_train_1 = accuracy.eval(feed_dict={X: train1_arr, y: train_y_1.reshape(-1)})
            acc_test = accuracy.eval(feed_dict={X: test_arr, y: test_y.reshape(-1)})
            cost = loss.eval(feed_dict={X: X_batch.reshape(20, -1), y: y_batch.reshape(-1)})
            accuracy_test.append(acc_test)

            summary, acc_train_00, loss_train = sess.run([merged, accuracy, loss], feed_dict={X: X_batch.reshape(20, -1), y: y_batch.reshape(-1)})
            train_writer.add_summary(summary, epoch)
            print("Epoch ", epoch, "Train_0 accuracy:", acc_train_0, "// Train_1 accuracy:", acc_train_1,
                  "// Test accuracy:", acc_test, "// Minibatch Loss :", cost)

            summary, acc_tests = sess.run([merged, accuracy], feed_dict={X: test_arr, y: test_y.reshape(-1)})
            test_writer.add_summary(summary, epoch)
        print("END")
        print("*" * 30)
    print("=" * 70)
    print("final result")
#############
# print(accuracy_test)
# print("Test accuracy: ", accuracy_test[-1])
    total_acc.append(accuracy_test[-1]) #

print(total_acc) #









