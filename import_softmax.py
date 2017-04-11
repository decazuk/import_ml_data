import tensorflow as tf
from input_data import read_data_sets

tf.set_random_seed(0)

train_data_set, test_data, test_label = read_data_sets()

weight_size = 6391
data_type_count = 2

X = tf.placeholder(tf.float32, [None, weight_size])
Y_ = tf.placeholder(tf.float32, [None, data_type_count])
W = tf.Variable(tf.zeros([weight_size, data_type_count]))
b= tf.Variable( tf.zeros([data_type_count]))

Y = tf.nn.softmax(tf.matmul(X, W) + b)

cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def training_step(i, update_train_data, update_test_data):
    batch_X, batch_Y = train_data_set.next_batch(100)
    if update_train_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y})
        print(str(i) + ": accuracy:" + str(a) + " loss:" + str(c))
    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: test_data.toarray(), Y_: test_label})
        print("   test accuracy:" + str(a) + " test loss:" + str(c))
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

for i in range(0, 15001):
    training_step(i, (i % 100) is 0, (i % 15000) is 0)

      
    