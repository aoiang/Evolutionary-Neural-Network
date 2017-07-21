import tensorflow as tf
filename_queue = tf.train.string_input_producer(["./resource/blah.txt"])

reader = tf.TextLineReader()

key, value = reader.read(filename_queue)
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.concat([[col1], [col2], [col3], [col4]], 0)

print(tf.shape(features))

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  example, label = sess.run([features, col5])
  print(example)








