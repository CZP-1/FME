from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

from resnet import *
from CBAM import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.util import random_noise
import random
from collections import Counter
from sklearn.model_selection import LeaveOneGroupOut
from scipy.signal import find_peaks
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d
random.seed(1)

import pickle

cp_callback=[]

def pseudo_labeling(final_images, final_samples, k):
    pseudo_y = []
    video_count = 0 
    
    for subject in final_samples:
        for video in subject:
            samples_arr = []
            if (len(video)==0):
                pseudo_y.append([0 for i in range(len(final_images[video_count])-k)]) #Last k frames are ignored
            else:
                pseudo_y_each = [0]*(len(final_images[video_count])-k)
                for ME in video:
                    samples_arr.append(np.arange(ME[0]+1, ME[1]+1))
                for ground_truth_arr in samples_arr: 
                    for index in range(len(pseudo_y_each)):
                        pseudo_arr = np.arange(index, index+k) 
                        # Equivalent to if IoU>0 then y=1, else y=0
                        if (pseudo_y_each[index] < len(np.intersect1d(pseudo_arr, ground_truth_arr))/len(np.union1d(pseudo_arr, ground_truth_arr))):
                            pseudo_y_each[index] = 1 
                pseudo_y.append(pseudo_y_each)
            video_count+=1
    
    # Integrate all videos into one dataset
    pseudo_y = [y for x in pseudo_y for y in x]
    print('Total frames:', len(pseudo_y))
    return pseudo_y
    
def loso(dataset, pseudo_y, final_images, final_samples, k):
    #To split the dataset by subjects
    y = np.array(pseudo_y)
    videos_len = []
    groupsLabel = y.copy()
    prevIndex = 0
    countVideos = 0
    
    #Get total frames of each video
    for video_index in range(len(final_images)):
      videos_len.append(final_images[video_index].shape[0]-k)
    
    print('Frame Index for each subject:-')
    for video_index in range(len(final_samples)):
      countVideos += len(final_samples[video_index])
      index = sum(videos_len[:countVideos])
      groupsLabel[prevIndex:index] = video_index
      print('Subject', video_index, ':', prevIndex, '->', index)
      prevIndex = index
    
    X = [frame for video in dataset for frame in video]
    print('\nTotal X:', len(X), ', Total y:', len(y))
    return X, y, groupsLabel
    
def normalize(images):
    for index in range(len(images)):
        for channel in range(3):
            images[index][:,:,channel] = cv2.normalize(images[index][:,:,channel], None, alpha=0, beta=1,norm_type=cv2.NORM_MINMAX)
    return images

def generator(X, y, batch_size=12, epochs=1):
    while True:
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            num_images = end - start
            X[start:end] = normalize(X[start:end])
            u = np.array(X[start:end])[:,:,:,0].reshape(num_images,42,42,1)
            v = np.array(X[start:end])[:,:,:,1].reshape(num_images,42,42,1)
            os = np.array(X[start:end])[:,:,:,2].reshape(num_images,42,42,1)
            yield [u, v, os], np.array(y[start:end])
            
def shuffling(X, y):
    shuf = list(zip(X, y))
    random.shuffle(shuf)
    X, y = zip(*shuf)
    return list(X), list(y)

def data_augmentation(X, y):
    transformations = {
        0: lambda image: np.fliplr(image), 
        1: lambda image: cv2.GaussianBlur(image, (7,7), 0),
        2: lambda image: random_noise(image),
    }
    y1=y.copy()
    for index, label in enumerate(y1):
        if (label==1): 
            for augment_type in range(3):
                img_transformed = transformations[augment_type](X[index]).reshape(42,42,3)
                X.append(np.array(img_transformed))
                y.append(1)
    return X, y

def SCBAM():
    inputs1 = layers.Input(shape=(42,42,1))
    conv1 = layers.Conv2D(3, (5,5), padding='same', activation='relu')(inputs1)
    pool1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv1)
    CBAM1 = cbam_module(pool1)
    # channel 2
    inputs2 = layers.Input(shape=(42,42,1))
    conv2 = layers.Conv2D(5, (5,5), padding='same', activation='relu')(inputs2)
    pool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv2)
    CBAM2 = cbam_module(pool2)
    # channel 3
    inputs3 = layers.Input(shape=(42,42,1))
    conv3 = layers.Conv2D(8, (5,5), padding='same', activation='relu')(inputs3)
    pool3 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv3)
    CBAM3 = cbam_module(pool3)

    # merge
    merged = layers.Concatenate()([CBAM1, CBAM2, CBAM3])
    
    # interpretation
    merged_pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(merged)

    flat = layers.Flatten()(merged_pool)
    dense = layers.Dense(400, activation='relu')(flat)
    outputs = layers.Dense(1, activation='linear')(dense)
    #Takes input u,v,s
    model = tf.keras.models.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    sgd = tf.keras.optimizers.SGD(lr=0.0005)   
    model.compile(loss="mse", optimizer=sgd, metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model

def SOFTNet():
    inputs1 = layers.Input(shape=(42,42,1))
    conv1 = layers.Conv2D(3, (5,5), padding='same', activation='relu')(inputs1)
    pool1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv1)
    # channel 2
    inputs2 = layers.Input(shape=(42,42,1))
    conv2 = layers.Conv2D(5, (5,5), padding='same', activation='relu')(inputs2)
    pool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv2)
    # channel 3
    inputs3 = layers.Input(shape=(42,42,1))
    conv3 = layers.Conv2D(8, (5,5), padding='same', activation='relu')(inputs3)
    pool3 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv3)
    # merge
    merged = layers.Concatenate()([pool1, pool2, pool3])
    # interpretation
    merged_pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(merged)
    flat = layers.Flatten()(merged_pool)
    dense = layers.Dense(400, activation='relu')(flat)
    outputs = layers.Dense(1, activation='linear')(dense)
    #Takes input u,v,s
    model = tf.keras.models.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    sgd = tf.keras.optimizers.SGD(lr=0.0005)   #?????????
    model.compile(loss="mse", optimizer=sgd, metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model

def spotting(result, total_gt, final_samples, subject_count, dataset, k, metric_fn, p=0.55):
    prev=0
    for videoIndex, video in enumerate(final_samples[subject_count-1]):
        preds = []
        gt = []
        countVideo = len([video for subject in final_samples[:subject_count-1] for video in subject])
        print('Video:', countVideo+videoIndex)
        score_plot = np.array(result[prev:prev+len(dataset[countVideo+videoIndex])]) #Get related frames to each video
        score_plot_agg = score_plot.copy()
        
        #Score aggregation
        for x in range(len(score_plot[k:-k])):
            score_plot_agg[x+k] = score_plot[x:x+2*k].mean()
        score_plot_agg = score_plot_agg[k:-k]
        plt.figure(figsize=(15,4))
        plt.plot(score_plot_agg) 
        plt.xlabel('Frame')
        plt.ylabel('Score')
        threshold = score_plot_agg.mean() + p * (max(score_plot_agg) - score_plot_agg.mean()) #Moilanen threshold technique
        peaks, _ = find_peaks(score_plot_agg[:,0], height=threshold[0], distance=k)
        if(len(peaks)==0): 
            preds.append([0, 0, 0, 0, 0, 0]) 
        for peak in peaks:
            preds.append([peak-k, 0, peak+k, 0, 0, 0]) 
        for samples in video:
            gt.append([samples[0]-k, 0, samples[1]-k, 0, 0, 0, 0])
            total_gt += 1
            plt.axvline(x=samples[0]-k, color='r')
            plt.axvline(x=samples[1]-k+1, color='r')    
            plt.axhline(y=threshold, color='g')
        plt.show()
        with open("micro-expression.txt",'a') as mae:
                mae.write(str(subject_count+5)+'_'+str(videoIndex+1)+'\n')
                mae.close()
        prev += len(dataset[countVideo+videoIndex])
        metric_fn.add(np.array(preds),np.array(gt)) 
    return preds, gt, total_gt
        
def evaluation(preds, gt, total_gt, metric_fn): #Get TP, FP, FN for final evaluation
    TP = int(sum(metric_fn.value(iou_thresholds=0.5)[0.5][0]['tp'])) 
    FP = int(sum(metric_fn.value(iou_thresholds=0.5)[0.5][0]['fp']))
    FN = total_gt - TP
    print('TP:', TP, 'FP:', FP, 'FN:', FN)
    return TP, FP, FN

def training(X, y, groupsLabel, dataset_name, expression_type, final_samples, k, dataset):
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X, y, groupsLabel)
    subject_count = 0
    epochs = 10
    batch_size = 12
    total_gt = 0
    metric_fn = MeanAveragePrecision2d(num_classes=1)
    p = 0.55 
    model = SCBAM()
    weight_reset = model.get_weights() #Initial weights

    for train_index, test_index in logo.split(X, y, groupsLabel): # Leave One Subject Out
        subject_count+=1
        print('Subject : ' + str(subject_count))
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index] #Get training set
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index] #Get testing set
        print('Dataset Labels', Counter(y_train))
        unique, uni_count = np.unique(y_train, return_counts=True) 
        rem_count = int(uni_count.max()*1/2)
        #Randomly remove non expression samples (With label 0) from dataset
        rem_index = random.sample([index for index, i in enumerate(y_train) if i==0], rem_count) 
        rem_index += (index for index, i in enumerate(y_train) if i>0)
        rem_index.sort()
        X_train = [X_train[i] for i in rem_index]
        y_train = [y_train[i] for i in rem_index]
        print('After Downsampling Dataset Labels', Counter(y_train))
        #Data augmentation to the micro-expression samples only
        if (expression_type == 'micro-expression'):
            X_train, y_train = data_augmentation(X_train, y_train)
            print('After Augmentation Dataset Labels', Counter(y_train))
            
        #Shuffle the training set
        X_train, y_train = shuffling(X_train, y_train)
        
        print('------Initializing -------') #To reset the model at every LOSO testing
        
        #path = 'SOFTNet_Weights/' + dataset_name + '/' + expression_type + '/s' + str(subject_count) + '.hdf5'
        model.set_weights(weight_reset) #Reset weights to ensure the model does not have info about current subject
        
        #model.load_model('path')  #Load Pretrained Weights
        path = 'Weights/' + dataset_name + '/' + expression_type + '/sb' + str(subject_count) + '.ckpt'
        path_model = 'Weights/' + dataset_name + '/' + expression_type + '/s' + str(subject_count) + '.h5'
        
        checkpoint_dir = os.path.dirname(path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                 save_weights_only=False,
                                                 verbose=1)
        
        model.fit(
            generator(X_train, y_train, batch_size, epochs),
            steps_per_epoch = len(X_train)/batch_size,
            epochs=epochs,
            verbose=1,
            validation_data = generator(X_test, y_test, batch_size),
            validation_steps = len(X_test)/batch_size,
            shuffle=True,
            callbacks=[cp_callback]
        )
       
        model.save(path_model)
        
        
        #model.load_weights(path)
        result = model.predict_generator(
            generator(X_test, y_test, batch_size),
            steps=len(X_test)/batch_size,
            verbose=1
        )
        
        preds, gt, total_gt = spotting(result, total_gt, final_samples, subject_count, dataset, k, metric_fn, p)
        TP, FP, FN = evaluation(preds, gt, total_gt, metric_fn)
        print(total_gt)
        
        print('Done Subject', subject_count)
    return TP, FP, FN, metric_fn

def final_evaluation(TP, FP, FN, metric_fn):
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1_score = (2 * precision * recall) / (precision + recall)
    
    print('TP:', TP, 'FP:', FP, 'FN:', FN)
    print('Precision = ', round(precision, 4))
    print('Recall = ', round(recall, 4))
    print('F1-Score = ', round(F1_score, 4))
    print("COCO AP@[.5:.95]:", round(metric_fn.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP'], 4))
