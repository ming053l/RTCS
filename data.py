
def setup_inputs(filenames, image_size=256, crop_size=224, isTest=False, batch_size=128):
    
    # Read each image file
    pdb.set_trace()
    filename_queue = tf.train.string_input_producer([filenames], num_epochs = 3)
    reader = tf.TFRecordReader()

    # read sample from TFRecords
    _, serialized_example = reader.read(filename_queue)

    # read one Example
    features = tf.parse_single_example(
    serialized_example,
    features={
      'image_raw': tf.FixedLenFeature([], tf.string),
      })

    image = tf.decode_raw(features['image_raw'], tf.float32)

    channels = 172
    image = tf.reshape(image, [image_size, image_size, channels])
        
    # Crop and other random augmentations
    if isTest is False:
#         image = tf.image.random_flip_left_right(image)
        image = tf.image.random_crop(image, [224, 224, 172])

    image = (tf.cast(image, tf.float32) - 4096.0)/4096.0
    numThr = 1 if isTest else 2
    image = tf.train.shuffle_batch([image], batch_size=batch_size, capacity=batch_size*2, num_threads=1, min_after_dequeue=2, name='hsi_images')

    return image
