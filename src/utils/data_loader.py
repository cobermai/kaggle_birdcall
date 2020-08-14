from __future__ import division
import numpy as np
import librosa 
import csv
import os
import glob
import warnings
import multiprocessing
import sys
import random
import copy
import time
import tensorflow as tf

warnings.filterwarnings('ignore')

class Dataset(object):
    """Implements the dataset properties
    """

    def __init__(self,path="",is_training_set=True):

        #Paths
        self.is_training_set = is_training_set
        self.path = path
        self.train_audio_path = os.path.join(path,"train_audio")
        self.train_csv_path = os.path.join(path,"train.csv")
        self.train_dict = self.csv_to_dict(self.train_csv_path)
        #Path of "false" audio samples
        self.false_audio_path = os.path.join(path,"false_audio")
        #Path to test audio
        self.test_audio_path = os.path.join(path,"example_test_audio")
        self.test_csv_path = os.path.join(path,"test.csv")
        self.test_dict = self.csv_to_dict(self.test_csv_path)
        self.test_meta_path = os.path.join(path,"example_test_audio_metadata.csv")
        self.test_dict = self.csv_to_dict(self.test_meta_path,self.test_dict)
        self.test_summary_path = os.path.join(path,"example_test_audio_summary.csv")
        self.test_dict = self.csv_to_dict(self.test_summary_path,self.test_dict)
        self.prepare()
        
    def csv_to_dict(self,path,data_dict = None):
        if data_dict == None:
            data_dict = {}
        with open(path, mode='r') as infile:
            reader = csv.reader(infile)
            first_row = True
            for rows in reader:
                if first_row:
                    col_names = rows
                    first_row = False
                    for name in col_names:
                        data_dict[name] = []
                col_ct = 0
                for name in col_names:
                    data_dict[name].append(rows[col_ct])
                    col_ct += 1
        return data_dict
        

    def prepare(self):
        """Prepares the Dataset class for use.
        """
        if self.is_training_set:
            #Prepare train samples 
            
            #Create bird mapping name->int
            if not(os.path.isfile(os.path.join(self.path,"bird_dict.npy"))):
                all_birds = self.train_dict['ebird_code']
                self.unique_birds_ebird_codes = np.unique(all_birds)
                self.bird_dict ={}
                bird_id = 0  
                for ebird_code in self.unique_birds_ebird_codes:
                    self.bird_dict[ebird_code] = bird_id
                    bird_id += 1
                
                
                np.save(os.path.join(self.path,"bird_dict.npy"),self.bird_dict)
            else:
                self.bird_dict = np.load(os.path.join(self.path,"bird_dict.npy"),allow_pickle=True).item()
            
            self.n_classes = len(self.bird_dict.keys())
            
            self.train_samples = []
            mp3_filenames = glob.glob(self.train_audio_path + "/**/*", 
                                recursive = True)
            for i_row in range(1,len(self.train_dict['filename'])):
                sample = {}
                for key in self.train_dict:
                    if len(self.train_dict[key])>i_row:
                        if key == 'filename':
                            search_name = self.train_dict[key][i_row]
                            for name in mp3_filenames:
                                if search_name in name:
                                    sample[key] = name
                                    break
                        elif key == 'ebird_code':
                            sample['bird_id'] = self.bird_dict[self.train_dict[key][i_row]]
                        else:
                            sample[key] = self.train_dict[key][i_row]
                    else:
                        sample[key] = None
                        
                
                self.train_samples.append(sample)
            self.n_samples = len(self.train_samples)
            
        else:
            #Prepare test samples
            self.test_samples = []
            try:
                self.bird_dict = np.load(os.path.join(self.path,"bird_dict.npy"),allow_pickle=True).item()
            except:
                raise("Run first with training set to create bird mapping!")
            
            mp3_filenames = glob.glob(self.test_audio_path + "/**/*", 
                                                    recursive = True)
            for i_row in range(1,len(self.test_dict['filename'])):
                sample = {}
                for key in self.test_dict:
                    if len(self.test_dict[key])>i_row:
                        if key == 'filename':
                            search_name = self.test_dict[key][i_row]
                            for name in mp3_filenames:
                                if search_name in name:
                                    sample[key] = name
                                    break
                        else:
                            sample[key] = self.test_dict[key][i_row]
                    else:
                        sample[key] = None
                        
                self.test_samples.append(sample)
            self.n_samples = len(self.test_samples)
        
class DataGenerator(object):
    def __init__(self,dataset,augmentation,
                 shuffle = True,
                 is_training = True,
                 is_validation = False,
                 force_feature_recalc = False,
                 preload_false_samples = True,
                 preload_samples = False,
                 training_percentage = 90,
                 save_created_features = True,
                 max_time = 5,
                 max_samples_per_audio = 6,
                 n_fft = 2048,
                 hop_length = 512,
                 sampling_rate = 22050):
        self.dataset = dataset
        #Shuffle files before loading since dataset is ordered by class
        if shuffle:
            random.seed(4)
            random.shuffle(self.dataset.train_samples)
        
        self.n_training_samples = int(dataset.n_samples*training_percentage/100)
        self.n_validation_samples = dataset.n_samples-self.n_training_samples
        self.augmentation = augmentation
        self.is_training = is_training
        self.is_validation = is_validation
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.preload_samples = preload_samples
        self.preload_false_samples = preload_false_samples
        self.hop_length = hop_length
        self.max_time = max_time
        self.max_samples_per_audio = max_samples_per_audio
        self.force_feature_recalc = force_feature_recalc
        self.save_created_features = save_created_features
        if self.is_training:
            self.first_sample = 0
            self.last_sample = self.n_training_samples
        elif self.is_validation:
            self.first_sample = self.n_training_samples
            self.last_sample = self.dataset.n_samples
        #Get paths of false samples
        false_samples_mono = glob.glob(self.dataset.false_audio_path+ "/mono/*.npz", 
                                        recursive = True)
        
        false_samples_stereo = glob.glob(self.dataset.false_audio_path+ "/stereo/*.npz", 
                                        recursive = True)
        
        self.false_sample_paths = false_samples_mono + false_samples_stereo
        #Pre load false samples
        if self.preload_false_samples:
            self.preloaded_false_samples = {}
            for path in self.false_sample_paths:
                with np.load(path,allow_pickle=True) as sample_file:
                    self.preloaded_false_samples[path] = sample_file.f.arr_0
            print("Finished pre-loading false samples!")
    
        
        if self.is_training or self.is_validation:
            self.samples = self.dataset.train_samples[self.first_sample:self.last_sample]
        else:
            self.samples = self.dataset.test_samples
        #Pre load samples (takes a lot of RAM ~130 GB)
        try:
            if self.preload_samples:
                self.preloaded_samples = {}
                for sample in self.samples:
                    path = sample["filename"].replace("mp3","npz")
                    with np.load(path,allow_pickle=True) as sample_file:
                        self.preloaded_samples[path] = sample_file.f.arr_0
                print("Finished pre-loading samples")
        except:
            self.preload_samples = False

    def do_stft(self,y,channels):
        spectra = []
        #STFT for all channels
        for channel in range(channels):
            spectrum = np.abs(librosa.core.stft(y[channel,:],
                                        n_fft      = self.n_fft,
                                        hop_length = self.hop_length,
                                        window     = 'hann', 
                                        center     = True))
            spectrum = np.asarray(spectrum,dtype=np.float32)
            spectra.append(spectrum)
        spectra = np.stack(spectra,axis=0)
        return spectra
    
    def pad_sample(self,spectrum,x_size=np.ceil(5*22050/512)):
        diff = int(x_size) - spectrum.shape[-1]
        if diff == 0:
            return spectrum
        if diff > spectrum.shape[-1]:
            while spectrum.shape[-1] < x_size:
                spectrum = np.concatenate([spectrum,spectrum],axis=-1)
                
            spectrum = spectrum[:,:,:int(x_size)]
        else:
            #First element is often zero. To avoid jump skip first element
            if diff+1 < spectrum.shape[-1]:
                spectrum = np.concatenate([spectrum,spectrum[:,:,1:diff+1]],axis=-1)
            else:
                spectrum = np.concatenate([spectrum,spectrum[:,:,:diff]],axis=-1)

        return spectrum
    
    def create_feature(self,sample):
        """Creates the features by doing a STFT"""

        filename = sample['filename']
        channels_str = sample['channels']
        channels = int(channels_str.split(" ")[0])
        if channels == 1:
            mono = True
        else:
            mono = False
        y, sr  = librosa.core.load(filename,mono=mono,sr=self.sampling_rate)
        
        y,_ = librosa.effects.trim(y)
        if mono == True:
            y = np.expand_dims(y,0)

        duration = y.shape[-1]/self.sampling_rate
        n_samples = int(np.ceil(duration/self.max_time))
        n_samples = min(n_samples,self.max_samples_per_audio)
        spectra = {}
        for i_sample in range(n_samples):
            start = i_sample*int(self.sampling_rate*self.max_time)
            end = (i_sample+1)*int(self.sampling_rate*self.max_time)
            end = min(end,y.shape[-1])
            y_sample = y[:,start:end]
            if y_sample.shape[-1] == 1:
                break
            #Transform audio
            spectrum = self.do_stft(y_sample,channels)
            #Pad spectrum
            spectrum = self.pad_sample(spectrum,
                                        x_size=np.ceil(self.max_time*self.sampling_rate/self.hop_length))
            spectra[str(i_sample)] = spectrum
            
        if self.save_created_features:
            if "mp3" in filename:
                np.savez(filename.replace("mp3","npz"),spectra)
            else:
                np.savez(filename.replace("wav","npz"),spectra)

        return spectra
    
    def create_all_features(self):
        if self.is_training:
            samples = self.dataset.train_samples
        else:
            samples = self.dataset.test_samples

        n = len(samples)

        ct = 0 
        for sample in samples:
            spectra = self.create_feature(sample)
            if np.any(spectra) == None:
                print(sample["filename"]+" failed!")
            else:
                print("Calculated "+str(ct/n*100)+"% of samples...")

            ct += 1 
            
    def create_all_features_multi_cpu(self):
        
        if self.is_training:
            all_samples = self.dataset.train_samples
        else:
            all_samples = self.dataset.test_samples
        
        samples = []
        for sample in all_samples:
            filename = sample['filename']
            if not(os.path.isfile(filename.replace("mp3","npz"))) or self.force_feature_recalc:
                samples.append(sample)
        
        print(str(len(all_samples)-len(samples))+" feature samples already exist")
        
        n = len(samples)
        
        pool = multiprocessing.Pool(os.cpu_count())
        for i, _ in enumerate(pool.imap_unordered(self.create_feature, samples), 1):
            sys.stderr.write('\rdone {0:%}'.format(max(0,i/n)))
        
    def create_false_features_multi_cpu(self):
        
        
        filenames_mono = glob.glob(self.dataset.false_audio_path+ "/mono/*.wav", 
                                        recursive = True)
        
        filenames_stereo = glob.glob(self.dataset.false_audio_path+ "/stereo/*.wav", 
                                        recursive = True)

        samples = []

        for filename in filenames_mono:
            if not(os.path.isfile(filename.replace("wav","npz"))) or self.force_feature_recalc:
                samples.append({'filename':filename,'channels':'1 mono'})


        for filename in filenames_stereo:
            if not(os.path.isfile(filename.replace("wav","npz"))) or self.force_feature_recalc:
                samples.append({'filename':filename,'channels':'2 stereo'})
                
        print(str(len(filenames_mono)+len(filenames_stereo)-len(samples))+" feature samples already exist")
        
        n = len(samples)

        pool = multiprocessing.Pool(os.cpu_count())
        for i, _ in enumerate(pool.imap_unordered(self.create_feature, samples), 1):
            sys.stderr.write('\rdone {0:%}'.format(max(0,i/n)))
            
    def generate_all_samples_from_scratch(self):
        stft_len = int(np.ceil(self.max_time*self.sampling_rate/self.hop_length))

        for sample in self.samples:

            filename = sample['filename']
            #Create features via STFT if no file exists
            spectra = self.create_feature(sample)
            
            for spec_key in spectra.keys():
                #Check for None type
                spectrum = spectra[spec_key]
                if np.any(spectrum) == None or spectrum.shape[-1] != stft_len:
                    continue
            
                #If only mono --> duplicate
                if spectrum.shape[0] == 1:
                    spectrum = np.tile(spectra[spec_key],[2,1,1])
                
                #Transpose spectrogramms for "channels_last"
                spectrum = tf.transpose(spectrum,perm=[1,2,0])
                
                #Fill false spectra with zero
                false_spectrum = tf.zeros_like(spectrum)
                
                if self.is_training or self.is_validation:
                    label = tf.one_hot(sample['bird_id'],self.dataset.n_classes+1)
                else:
                    label = None

                sub_sample = {'input_features':spectrum,
                    'labels':label,
                    'false_sample':false_spectrum}
                
                if self.augmentation != None:
                    yield self.augmentation(sub_sample,self.is_training)
                else:
                    yield sample
        
    def generate(self):
        
        
        stft_len = int(np.ceil(self.max_time*self.sampling_rate/self.hop_length))

        for sample in self.samples:

            filename = sample['filename']
            #If feature was already created load from file
            if os.path.isfile(filename.replace("mp3","npz")) and not(self.force_feature_recalc):
                if self.preload_samples:
                    spectra_npz = self.preloaded_samples[filename.replace("mp3","npz")]
                else:
                    with np.load(filename.replace("mp3","npz"),allow_pickle=True) as sample_file:
                        spectra_npz = sample_file.f.arr_0
                
                spec_keys = spectra_npz.item().keys()
                spec_keys = list(spec_keys)
                rnd_key = spec_keys[np.random.randint(0,len(spec_keys))]
                spectra = spectra_npz.item()[rnd_key]
            else:
                #Create features via STFT if no file exists
                spectra = self.create_feature(sample)
                spec_keys = spectra.keys()
                spec_keys = list(spec_keys)
                rnd_key = spec_keys[np.random.randint(0,len(spec_keys))]
                spectra = spectra[rnd_key]

            #Check for None type
            if np.any(spectra) == None or spectra.shape[-1] != stft_len:
                continue
            
            #Get false sample
            rnd_false_sample = random.choice(self.false_sample_paths)

            if self.preload_false_samples:
                false_spectra_npz = self.preloaded_false_samples[rnd_false_sample]
            else:
                with np.load(rnd_false_sample,allow_pickle=True) as sample_file:
                    false_spectra_npz = sample_file.f.arr_0
                
            false_spec_keys = false_spectra_npz.item().keys()
            false_spec_keys = list(false_spec_keys)
            false_rnd_key = false_spec_keys[np.random.randint(0,len(false_spec_keys))]
            false_spectra = false_spectra_npz.item()[false_rnd_key]

            #If only mono --> duplicate
            if spectra.shape[0] == 1:
                spectra = np.tile(spectra,[2,1,1])
                
            
            #If false only mono --> duplicate
            if false_spectra.shape[0] == 1:
                false_spectra = np.tile(false_spectra,[2,1,1])

            #Transpose spectrogramms for "channels_last"
            spectra = tf.transpose(spectra,perm=[1,2,0])
            false_spectra = tf.transpose(false_spectra,perm=[1,2,0])

            sample = {'input_features':spectra,
                'labels':tf.one_hot(sample['bird_id'],self.dataset.n_classes+1),
                'false_sample':false_spectra}
            if self.augmentation != None:
                yield self.augmentation(sample,self.is_training)
            else:
                yield sample

if __name__ == "__main__":
    ds = Dataset("/srv/TUG/datasets/cornell_birdcall_recognition")
    dg = DataGenerator(ds,None,force_feature_recalc=True)
    dg.create_all_features_multi_cpu()
