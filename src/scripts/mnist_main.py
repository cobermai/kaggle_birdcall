import os
import numpy as np
from absl import logging
from absl import app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order
from utils.trainer import ModelTrainer
from models import basic_dmoe_cnn

#logging.set_verbosity(logging.WARNING)

HEIGHT = 28 
WIDTH = 28
NUM_CHANNELS = 1
BATCH_SIZE = 64
NUM_CLASSES = 10
_NUM_DATA_FILES = 5

# TODO(tobyboyd): Change to best practice 45K(train)/5K(val)/10K(test) splits.
NUM_IMAGES = {
    'train': 55000,
    'validation': 5000,
    'test': 10000,
}

DATASET_NAME = 'MNIST'



def preprocess_image(image):
    """Preprocess a single image of layout [height, width, depth]."""
    return tf.expand_dims(image/255,2)


def data_generator(data,batch_size,is_training,is_validation=False,take_n=None,skip_n=None):
   
    dataset = tf.data.Dataset.from_tensor_slices(data)
    if is_training:
        shuffle_buffer=NUM_IMAGES['train']
    elif is_validation:
        shuffle_buffer=NUM_IMAGES['validation']
    else:
        shuffle_buffer=NUM_IMAGES['test']

    if skip_n != None:
        dataset = dataset.skip(skip_n)
    if take_n != None:
        dataset = dataset.take(take_n)

    if is_training:

        dataset = dataset.shuffle(shuffle_buffer)
        #dataset = dataset.batch(1000) # Batch here so that resize_with_crop_or_pad works more efficient
        #dataset = dataset.map(lambda img, lbl: (tf.image.resize_with_crop_or_pad(img, HEIGHT + 8, WIDTH + 8), lbl))
        #dataset = dataset.unbatch()
        dataset = dataset.map(lambda img, lbl: (preprocess_image(img), lbl))
        dataset = dataset.map(lambda img, lbl: (img, tf.one_hot(lbl,NUM_CLASSES)))
        dataset = dataset.batch(BATCH_SIZE,drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(lambda img, lbl: (preprocess_image(img), lbl))
        dataset = dataset.map(lambda img, lbl: (img, tf.one_hot(lbl,NUM_CLASSES)))
        dataset = dataset.batch(100)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def learning_rate_fn(epoch):

    if epoch >= 20 and epoch <30:
        return 0.01
    elif epoch >=30 and epoch <40:
        return 0.001
    elif epoch >=40:
        return 0.001
    else:
        return 1.0


FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', '/tmp', 'save directory name')
flags.DEFINE_string('data_dir', '/tmp', 'data directory name')
flags.DEFINE_integer('epochs', 40, 'number of epochs')

flags.DEFINE_float('dropout_rate', 0.0, 'dropout rate for the dense blocks')
flags.DEFINE_float('weight_decay', 1e-4, 'weight decay parameter')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')

flags.DEFINE_boolean('load_model', False, 'Bool indicating if the model should be loaded')



def main(argv):
    
    try:

        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

    except KeyError:

        task_id = 0
    
    
    model_save_dir = FLAGS.model_dir
    data_dir = FLAGS.data_dir
    print("Saving model to : " + str(model_save_dir))
    print("Loading data from : " + str(data_dir))
    test_data_dir = data_dir
    train_data_dir = data_dir
    epochs = FLAGS.epochs
    dropout_rate = FLAGS.dropout_rate
    weight_decay = FLAGS.weight_decay
    lr = FLAGS.learning_rate
    load_model = FLAGS.load_model

    model_save_dir+="_dropout_rate_"+str(dropout_rate)+"_learning_rate_"+str(lr)+"_weight_decay_"+str(weight_decay)

    model = basic_dmoe_cnn.basic_dmoe_cnn_mnist()

    train_data, test_data = tf.keras.datasets.mnist.load_data()

    train_data_gen = data_generator(train_data,BATCH_SIZE,is_training=True,take_n=NUM_IMAGES["train"])
    val_data_gen = data_generator(train_data,100,is_training=False,is_validation = True,skip_n=NUM_IMAGES["train"],take_n=NUM_IMAGES["validation"])
    test_data_gen = data_generator(test_data,100,is_training=False)

    trainer = ModelTrainer(model,
                    train_data_gen,
                    val_data_gen,
                    test_data_gen,
                    epochs,
                    learning_rate_fn = learning_rate_fn,
                    optimizer = tf.keras.optimizers.Adam,
                    num_train_batches = int(NUM_IMAGES["train"]/BATCH_SIZE),
                    base_learning_rate = lr,
                    load_model = load_model,
                    save_dir = model_save_dir,
                    init_data = tf.random.normal([BATCH_SIZE,28,28,1]),
                    start_epoch = 0)
    
    trainer.train()

if __name__ == '__main__':
  app.run(main)
  
