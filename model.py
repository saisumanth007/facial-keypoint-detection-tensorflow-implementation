import tensorflow as tf
from utilities import *

train_dataset, train_labels = load_data()
test_dataset, _ = load_data(test = True)

#Validation Dataset
validation_dataset = train_dataset[:VALIDATION_SIZE,...]
validation_labels = train_labels[:VALIDATION_SIZE]
train_dataset = train_dataset[VALIDATION_SIZE:,...]
train_labels = train_labels[VALIDATION_SIZE:]

train_input = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
train_output = tf.placeholder(tf.float32, shape=(BATCH_SIZE,LABELS_SIZE))

eval_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

train_size = train_labels.shape[0]

# conv1_weights = tf.get_variable("conv1_weights", shape=[5,5,1,32], initializer=tf.contrib.layers.xavier_initializer())
# conv1_biases = tf.Variable(tf.zeros([32]))

# conv2_weights = tf.get_variable("conv2_weights", shape=[5,5,32,64], initializer=tf.contrib.layers.xavier_initializer())
# conv2_biases = tf.Variable(tf.zeros([64]))

# fc1_weights = tf.get_variable("fc1_weights", shape=[IMAGE_SIZE//4 * IMAGE_SIZE//4 * 64, 512], initializer=tf.contrib.layers.xavier_initializer())
# fc1_biases = tf.Variable(tf.constant(0.1,shape = [512]))

# fc2_weights = tf.get_variable("fc2_weights", shape=[512, LABELS_SIZE], initializer=tf.contrib.layers.xavier_initializer())
# fc2_biases = tf.Variable(tf.constant(0.1,shape = [LABELS_SIZE]))


conv1_weights = tf.Variable(
    tf.truncated_normal([5, 5, NUM_CHANNELS, 32],
                        stddev=0.1,
                        seed=SEED))
conv1_biases = tf.Variable(tf.zeros([32]))

conv2_weights = tf.Variable(
    tf.truncated_normal([5, 5, 32, 64],
                        stddev=0.1,
                        seed=SEED))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

fc1_weights = tf.Variable(  # fully connected, depth 512.
                            tf.truncated_normal(
                                [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                                stddev=0.1,
                                seed=SEED))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))

fc2_weights = tf.Variable(  # fully connected, depth 512.
                            tf.truncated_normal(
                                [512, 512],
                                stddev=0.1,
                                seed=SEED))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[512]))

fc3_weights = tf.Variable(  # fully connected, depth 512.
                            tf.truncated_normal(
                                [512, LABELS_SIZE],
                                stddev=0.1,
                                seed=SEED))
fc3_biases = tf.Variable(tf.constant(0.1, shape=[LABELS_SIZE]))

def model(data, train = False):
	#data = tf.reshape(data,[-1,IMAGE_SIZE,IMAGE_SIZE,1])
	conv = tf.nn.conv2d(data, conv1_weights, strides = [1,1,1,1], padding = 'SAME')
	relu = tf.nn.relu(tf.nn.bias_add(conv,conv1_biases))
	pool = tf.nn.max_pool(relu, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

	conv = tf.nn.conv2d(pool, conv2_weights, strides = [1,1,1,1], padding = 'SAME')
	relu = tf.nn.relu(tf.nn.bias_add(conv,conv2_biases))
	pool = tf.nn.max_pool(relu, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

	pool_shape = pool.get_shape().as_list()

	pool_flat = tf.reshape(pool,[pool_shape[0],pool_shape[1]*pool_shape[2]*pool_shape[3]])

	hidden = tf.nn.relu(tf.matmul(pool_flat,fc1_weights) + fc1_biases)
	if train:
		hidden = tf.nn.dropout(hidden, 0.5)
	hidden = tf.nn.relu(tf.matmul(hidden,fc2_weights) + fc2_biases)
	if train:
		hidden = tf.nn.dropout(hidden, 0.5)
	return tf.matmul(hidden,fc3_weights) + fc3_biases

predictions = model(train_input, True)

cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(predictions-train_output),1))
regularizers = (tf.nn.l2_loss(fc1_weights)+tf.nn.l2_loss(fc2_weights))
cross_entropy += 1e-7 * regularizers

eval_prediction = model(eval_data_node)

global_step = tf.Variable(0,trainable = False)

learning_rate = tf.train.exponential_decay(
	1e-3,
	global_step * BATCH_SIZE,
	train_size,
	0.95,
	staircase = True
	)

train_step = tf.train.AdamOptimizer(learning_rate,0.95).minimize(cross_entropy, global_step = global_step)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

best_valid = np.inf
best_valid_epoch = 0
current_epoch = 0

while current_epoch < NUM_EPOCHS:
	print 'Epoch "%04d" running '%(current_epoch)
	
	shuffled_index = np.arange(train_size)
	np.random.shuffle(shuffled_index)
	train_dataset = train_dataset[shuffled_index]
	train_labels = train_labels[shuffled_index]

	for step in xrange(train_size/BATCH_SIZE):
		offset = step * BATCH_SIZE
		batch_data = train_dataset[offset:(offset+BATCH_SIZE),...]
		batch_labels = train_labels[offset:(offset+BATCH_SIZE)]

		feed_dict = {train_input:batch_data, train_output:batch_labels}
		_, loss_train, current_learning_rate = sess.run([train_step,cross_entropy,learning_rate], feed_dict = feed_dict)

	eval_result = eval_in_batches(validation_dataset, sess, eval_prediction, eval_data_node)
	loss_valid = error_measure(eval_result, validation_labels)

	print 'Epoch %04d, train loss %.8f, validation loss %.8f, learning rate %0.8f' % (
            current_epoch,
            loss_train, loss_valid,
            current_learning_rate
            )

	if loss_valid < best_valid:
		best_valid = loss_valid
		best_valid_epoch = current_epoch

	elif best_valid_epoch + EARLY_STOP_PATIENCE < current_epoch:
		print "Early stopping"
		print "Best valid loss was {:.6f} at epoch {}.".format(best_valid, best_valid_epoch)
		break

	current_epoch += 1

print "Train finish"
generate_submission(test_dataset, sess, eval_prediction, eval_data_node)