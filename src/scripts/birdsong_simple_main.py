import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import librosa
from absl import logging
from absl import app
from absl import flags
from utils.trainer import ModelTrainer
from utils.data_loader import Dataset
from utils.data_loader import DataGenerator
from utils.summary_utils import Summaries
from models.classifier import Classifier
from models.res_block import ResBlockBasicLayer
from models.discriminator import Discriminator
from models.generator import Generator
from models.eval_functions.separate_classifier_wgan_eval_fns import EvalFunctions
import tensorflow as tf  # pylint: disable=g-bad-import-order

BINS = 1025
N_FRAMES = 216
N_CHANNELS = 2


def augment_input(sample,n_classes,training):
    """Preprocess a single image of layout [height, width, depth]."""
    
    input_features = sample['input_features']
    labels = sample['labels']
    false_sample = sample['false_sample']
    if training:
        rnd = tf.random.uniform([1])
        rnd_tiled_feat = tf.tile(tf.reshape(rnd,[1,1,1]),[BINS,
                                                    N_FRAMES,
                                                    N_CHANNELS])

        rnd_tiled_lbl = tf.tile(tf.reshape(rnd,[1]),[n_classes+1])

        false_lbl = tf.cast(tf.one_hot(n_classes+1,n_classes+1),labels.dtype)

        input_features = tf.where(rnd_tiled_feat > 1/n_classes,input_features,false_sample)

        labels = tf.where(rnd_tiled_lbl > 1/n_classes,labels,false_lbl)

    input_features = tf.image.per_image_standardization(input_features)
    false_sample = tf.image.per_image_standardization(false_sample)

    return {'input_features':input_features,'labels':labels,'false_sample':false_sample}

def data_generator(data_generator,batch_size,is_training,
                   shuffle_buffer = 128,
                   is_validation=False,
                   n_classes = 10,
                   take_n=None,
                   skip_n=None):
   
    dataset = tf.data.Dataset.from_generator(data_generator,
                                             output_types = {'input_features':tf.float32,
                                                             'labels':tf.int32,
                                                             'false_sample':tf.float32})

    if skip_n != None:
        dataset = dataset.skip(skip_n)
    if take_n != None:
        dataset = dataset.take(take_n)
    
    if is_training:
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(batch_size,drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def learning_rate_fn(epoch):

    if epoch >= 150 and epoch <200:
        return 0.1
    elif epoch >=200 and epoch <250:
        return 0.01
    elif epoch >=250:
        return 0.001
    else:
        return 1.0


FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', 'models/saved_models', 'save directory name')
flags.DEFINE_string('data_dir', 'data/cornell_birdcall_recognition_mini', 'data directory name')
flags.DEFINE_integer('epochs', 300, 'number of epochs')
flags.DEFINE_integer('batch_size', 32, 'Mini-batch size')
flags.DEFINE_float('dropout_rate', 0.0, 'dropout rate for the dense blocks')
flags.DEFINE_float('weight_decay', 1e-4, 'weight decay parameter')
flags.DEFINE_float('learning_rate', 5e-2, 'learning rate')
flags.DEFINE_boolean('preload_samples',False,'Preload samples (requires >140 GB RAM)')
flags.DEFINE_float('training_percentage', 90, 'Percentage of the training data used for training. (100-training_percentage is used as validation data.)')

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
    batch_size = FLAGS.batch_size
    dropout_rate = FLAGS.dropout_rate
    weight_decay = FLAGS.weight_decay
    lr = FLAGS.learning_rate
    load_model = FLAGS.load_model
    training_percentage = FLAGS.training_percentage
    preload_samples = FLAGS.preload_samples

    model_save_dir += "_batch_size_"+str(batch_size)+"_dropout_rate_"+str(dropout_rate)+"_learning_rate_"+str(lr)+"_weight_decay_"+str(weight_decay)

    ds_train = Dataset(data_dir,is_training_set = True)
    n_total = ds_train.n_samples
    
    def augment_fn(sample,training):
        return augment_input(sample,ds_train.n_classes,training)
    
    dg_train = DataGenerator(ds_train,augment_fn,
                             training_percentage = training_percentage,
                             preload_samples = preload_samples,
                             is_training=True)
    
    dg_val = DataGenerator(ds_train,augment_fn,
                           training_percentage = training_percentage,
                           is_validation = True,
                           preload_samples = preload_samples,
                           is_training=False)
    n_train = int(n_total*training_percentage/100)
    n_val = n_total-n_train

    #ResNet 18
    classifier_model = Classifier(ResBlockBasicLayer,
                 n_blocks = 4,
                 n_layers = [2,2,2,2],
                 strides = [2,2,2,2],
                 channel_base = [64,128,256,512],
                 n_classes = ds_train.n_classes+1,
                 init_ch = 64,
                 init_ksize = 7,
                 init_stride = 2,
                 use_max_pool = True,
                 kernel_regularizer = tf.keras.regularizers.l2(2e-4),
                 kernel_initializer = tf.keras.initializers.he_normal(),
                 name = "classifier",
                 dropout=dropout_rate)
    #Generator model used to augment to false samples
    generator_model = Generator(8,
                                [8,8,16,16,32,32,64,64],
                                kernel_regularizer = tf.keras.regularizers.l2(2e-4),
                                kernel_initializer = tf.keras.initializers.he_normal(),
                                name = "generator")
    
    #Discriminator for estimating the Wasserstein distance
    discriminator_model = Discriminator(3,
                                        [32,64,128],
                                        [4,4,4],
                                        name = "discriminator")
    
    train_data_gen = data_generator(dg_train.generate,batch_size,
                                    is_training=True,
                                    shuffle_buffer = 16*batch_size,
                                    n_classes = ds_train.n_classes,
                                    take_n=n_train)
    
    val_data_gen = data_generator(dg_val.generate,batch_size,
                                  is_training=False,
                                  is_validation = True,
                                  n_classes = ds_train.n_classes)
    
    
    summaries = Summaries(scalar_summary_names=["class_loss",
                                      "generator_loss",
                                      "discriminator_loss",
                                      "weight_decay_loss",
                                      "total_loss",
                                      "accuracy"],
                              image_summary_settings = {'train':['fake_features'],'n_images':1},
                              learning_rate_names = ['learning_rate_'+str(classifier_model.model_name),
                                      'learning_rate_'+str(generator_model.model_name),
                                      'learning_rate_'+str(discriminator_model.model_name)],
                              save_dir = model_save_dir,
                              modes = ["train","val","test"],
                              summaries_to_print={'train':['class_loss','generator_loss','discriminator_loss','accuracy'],
                                                'eval':['total_loss','accuracy']})
    

    trainer = ModelTrainer(train_data_gen,
                    val_data_gen,
                    None,
                    epochs,
                    EvalFunctions,
                    model_settings = [{'model':classifier_model,
                               'optimizer_type':tf.keras.optimizers.SGD,
                               'base_learning_rate':lr,
                               'learning_rate_fn':learning_rate_fn,
                               'init_data':tf.random.normal([batch_size,BINS,N_FRAMES,N_CHANNELS])},
                              {'model':generator_model,
                               'optimizer_type':tf.keras.optimizers.Adam,
                               'base_learning_rate':lr*0.01,
                               'learning_rate_fn':learning_rate_fn,
                               'init_data':tf.random.normal([batch_size,BINS,N_FRAMES,N_CHANNELS])},
                              {'model':discriminator_model,
                               'optimizer_type':tf.keras.optimizers.Adam,
                               'base_learning_rate':lr*0.02,
                               'learning_rate_fn':learning_rate_fn,
                               'init_data':tf.random.normal([batch_size,BINS,N_FRAMES,N_CHANNELS])}],
                    summaries = summaries,
                    num_train_batches = int(n_train/batch_size),
                    load_model = load_model,
                    save_dir = model_save_dir,
                    input_keys = ["input_features","false_sample"],
                    label_keys = ["labels"],
                    init_data = tf.random.normal([batch_size,BINS,N_FRAMES,N_CHANNELS]),
                    start_epoch = 0)
    
    trainer.train()

if __name__ == '__main__':
  app.run(main)
  
