from __future__ import print_function, division

from keras.datasets import mnist
import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import  Conv2D
from keras.layers import UpSampling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.applications import ResNet50
from keras.engine import base_layer
from keras import layers
from keras.utils import to_categorical
import time
from keras.utils import generic_utils
import keras.backend as K
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from data_loader import DataLoader
from model.frcnn_model import FRCNN
from tools import roi_helpers
import sys
import os
import numpy as np
from cfg.config import Config
from model import losses as losses_fn
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
path = "../wgan_sn_info"
if not os.path.exists(path):
    os.mkdir(path)

ouput_images_path=path+"/images"
if not os.path.exists(ouput_images_path):
    os.mkdir(ouput_images_path)
saved_model_path=path+"/saved_model"
if not os.path.exists(saved_model_path):
    os.mkdir(saved_model_path)
saved_log=path+"/saved_log"
if not os.path.exists(saved_log):
    os.mkdir(saved_log)

class MyRcnn():
    def __init__(self):
        self.cfg = Config()
        self.cfg.model_path=saved_model_path+'/kitti_frcnn_last.hdf5'
        frcnn=FRCNN(self.cfg.num_classes,self.cfg.num_anchors)
        num_anchors = len(self.cfg.anchor_box_scales) * len(self.cfg.anchor_box_ratios)
        #self.data_loader = DataLoader(dataset_name=self.dataset_name,
        #                              img_res= self.img_shape)

        # Following parameter and optimizer set as recommended in paper
        optimizer = Adam(0.0002, 0.5)

        input_shape_img = (None, None, 3)
        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(None, 4))

        self.pretrain = frcnn.build_share()
        shared_layers=self.pretrain(img_input)

        #buid rpn
        rpn_out = frcnn.build_rpn(shared_layers)
        self.rpn = Model(img_input,rpn_out)

        #buid classify
        cla_out =frcnn.build_classifier(shared_layers,roi_input)
        self.classify = Model([img_input,roi_input],cla_out)

        #build combined

        self.combined=Model([img_input, roi_input],rpn_out+cla_out)

        #compile

        self.rpn.compile(optimizer=optimizer,
                      loss=[losses_fn.rpn_loss_cls(num_anchors), losses_fn.rpn_loss_regr(num_anchors)])
        self.classify.compile(optimizer=optimizer,
                            loss=[losses_fn.class_loss_cls, losses_fn.class_loss_regr(10 - 1)],
                            metrics={'dense_class_{}'.format(10): 'accuracy'})

        self.combined.compile(optimizer='sgd', loss='mae')
        self.data_loader = DataLoader(self.cfg,
                                      img_res=(512,512,3))


    def save_imgs(self, epoch):
        # r, c = 10, 10
        # fig, axs = plt.subplots(r, c)
        # for i in range(c):
        #     sampled_noise, _ = self.sample_generator_input(c)
        #     label = to_categorical(np.full(fill_value=i, shape=(r,1)), num_classes=self.classes)
        #     gen_input = np.concatenate((sampled_noise, label), axis=1)
        #     gen_imgs = self.generator.predict(gen_input)
        #     gen_imgs = 0.5 * gen_imgs + 0.5
        #     for j in range(r):
        #         if self.channels>1:
        #             axs[i,j].imshow(gen_imgs[j])
        #         else:
        #             axs[i,j].imshow(gen_imgs[j,:,:,0],"gray")
        #         axs[i,j].axis('off')
        # fig.savefig(ouput_images_path+"/example_%d.png" % epoch)
        # plt.close()
        return 0
    def save_model(self):
        # def save(model, model_name):
        #     model_path = saved_model_path +"/%s.json" % model_name
        #     weights_path = saved_model_path +"/%s_weights.h5" % model_name
        #     options = {"file_arch": model_path,
        #                 "file_weight": weights_path}
        #     json_string = model.to_json()
        #     open(options['file_arch'], 'w').write(json_string)
        #     model.save_weights(options['file_weight'])

        # save(self.generator, "generator")
        # save(self.critic, "discriminator")
        # save(self.classify, "classify")
        return 0
    def save_log(self):
        # d_loss=np.array(self.d_loss)
        # np.save(saved_log+"/d_loss.npy",d_loss)
        # # d_acc=np.array(self.acc)
        # # np.save(saved_log+"/d_acc.npy",d_acc)
        # g_loss=np.array(self.g_loss)
        # np.save(saved_log+"/g_loss.npy",g_loss)
        # c_loss=np.array(self.c_loss)
        # np.save(saved_log+"/c_loss.npy",c_loss)
        return 0
    def plot_log(self):
        # r,c = 3,1 
        # def sub_plt(data,idx,x_label,y_label,color):
        #     plt.subplot(r,c,idx)
        #     plt.xlabel(x_label)
        #     plt.ylabel(y_label)
        #     plt.plot(data,color=color)
        # fig=plt.figure(figsize=(16,12),dpi=200)
        # plt.title("loss")
        # sub_plt(self.d_loss,1,"iter","d_loss","blue")
        # sub_plt(self.g_loss,2,"iter","g_loss","red")
        # sub_plt(self.c_loss,3,"iter","c_loss","green")
        # fig.savefig(saved_log+"/log.png" )
        # plt.close()
        return 0
    def sample_generator_input(self, batch_size):
        # Generator inputs
        # sampled_noise = np.random.normal(0, 1, (batch_size, 62))
        # sampled_labels = np.random.randint(0, self.classes, batch_size).reshape(-1, 1)
        # sampled_labels = to_categorical(sampled_labels, num_classes=self.classes)

        # return sampled_noise, sampled_labels
        return 0
    def load_train(self):
        # if os.path.exists(saved_model_path+"/discriminator_weights.h5"):
        #     self.critic.load_weights(saved_model_path+"/discriminator_weights.h5")
        #     print("load generator weights")
        # if os.path.exists(saved_model_path+"/generator_weights.h5"):
        #     self.generator.load_weights(saved_model_path+"/generator_weights.h5")
        #     print("load discriminator weights")
        # if os.path.exists(saved_model_path+"/classify_weights.h5"):
        #     self.classify.load_weights(saved_model_path+"/classify_weights.h5")
        #     print("load classify weights")
        # if os.path.exists(saved_log+"/g_loss.npy"):
        #     self.g_loss=(np.load(saved_log+"/g_loss.npy")).tolist()
        #     self.d_loss=(np.load(saved_log+"/d_loss.npy")).tolist()
        #     self.c_loss=(np.load(saved_log+"/c_loss.npy")).tolist()
        #     self.ite=len(self.g_loss)
        #     print("load log")
        return 0
    def train(self, epochs):
        data_gen_train = self.data_loader.get_anchor_gt(mode='train')
        epoch_length = 1000
        num_epochs = epochs
        iter_num=0
        losses = np.zeros((epoch_length, 5))
        rpn_accuracy_rpn_monitor = []
        rpn_accuracy_for_epoch = []
        start_time = time.time()
        best_loss = np.Inf

        for epoch_num in range(num_epochs):
            progbar = generic_utils.Progbar(epoch_length)
            print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
            while True:
                try:
                    if len(rpn_accuracy_rpn_monitor) == epoch_length and self.cfg.verbose:
                        mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                        rpn_accuracy_rpn_monitor = []
                        print(
                            'Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                                mean_overlapping_bboxes, epoch_length))
                        if mean_overlapping_bboxes == 0:
                            print('RPN is not producing bounding boxes that overlap'
                                ' the ground truth boxes. Check RPN settings or keep training.')

                    X, Y, img_data = next(data_gen_train)

                    loss_rpn = self.rpn.train_on_batch(X, Y)

                    P_rpn = self.rpn.predict_on_batch(X)

                    result = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], self.cfg, K.image_dim_ordering(), use_regr=True,
                                                    overlap_thresh=0.7,
                                                    max_boxes=300)
                    # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                    X2, Y1, Y2, IouS = roi_helpers.calc_iou(result, img_data, self.cfg, self.data_loader.class_mapping)

                    if X2 is None:
                        rpn_accuracy_rpn_monitor.append(0)
                        rpn_accuracy_for_epoch.append(0)
                        continue

                    neg_samples = np.where(Y1[0, :, -1] == 1)
                    pos_samples = np.where(Y1[0, :, -1] == 0)

                    if len(neg_samples) > 0:
                        neg_samples = neg_samples[0]
                    else:
                        neg_samples = []

                    if len(pos_samples) > 0:
                        pos_samples = pos_samples[0]
                    else:
                        pos_samples = []

                    rpn_accuracy_rpn_monitor.append(len(pos_samples))
                    rpn_accuracy_for_epoch.append((len(pos_samples)))

                    if self.cfg.num_rois > 1:
                        if len(pos_samples) < self.cfg.num_rois // 2:
                            selected_pos_samples = pos_samples.tolist()
                        else:
                            selected_pos_samples = np.random.choice(pos_samples, self.cfg.num_rois // 2, replace=False).tolist()
                        try:
                            selected_neg_samples = np.random.choice(neg_samples, self.cfg.num_rois - len(selected_pos_samples),
                                                                    replace=False).tolist()
                        except:
                            selected_neg_samples = np.random.choice(neg_samples, self.cfg.num_rois - len(selected_pos_samples),
                                                                    replace=True).tolist()

                        sel_samples = selected_pos_samples + selected_neg_samples
                    else:
                        # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                        selected_pos_samples = pos_samples.tolist()
                        selected_neg_samples = neg_samples.tolist()
                        if np.random.randint(0, 2):
                            sel_samples = random.choice(neg_samples)
                        else:
                            sel_samples = random.choice(pos_samples)

                    loss_class = self.classify.train_on_batch([X, X2[:, sel_samples, :]],
                                                                [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                    losses[iter_num, 0] = loss_rpn[1]
                    losses[iter_num, 1] = loss_rpn[2]

                    losses[iter_num, 2] = loss_class[1]
                    losses[iter_num, 3] = loss_class[2]
                    losses[iter_num, 4] = loss_class[3]

                    iter_num += 1

                    progbar.update(iter_num,
                                [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                    ('detector_cls', np.mean(losses[:iter_num, 2])),
                                    ('detector_regr', np.mean(losses[:iter_num, 3]))])

                    if iter_num == epoch_length:
                        loss_rpn_cls = np.mean(losses[:, 0])
                        loss_rpn_regr = np.mean(losses[:, 1])
                        loss_class_cls = np.mean(losses[:, 2])
                        loss_class_regr = np.mean(losses[:, 3])
                        class_acc = np.mean(losses[:, 4])

                        mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                        rpn_accuracy_for_epoch = []

                        if self.cfg.verbose:
                            print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                                mean_overlapping_bboxes))
                            print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                            print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                            print('Loss RPN regression: {}'.format(loss_rpn_regr))
                            print('Loss Detector classifier: {}'.format(loss_class_cls))
                            print('Loss Detector regression: {}'.format(loss_class_regr))
                            print('Elapsed time: {}'.format(time.time() - start_time))

                        curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                        iter_num = 0
                        start_time = time.time()

                        if curr_loss < best_loss:
                            if self.cfg.verbose:
                                print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                            best_loss = curr_loss
                            self.combined.save_weights(self.cfg.model_path)

                        break

                except Exception as e:
                    print('Exception: {}'.format(e))
                    # save model
                    self.combined.save_weights(self.cfg.model_path)
                    continue            
        # # Adversarial ground truths
        # (X_train, y_train), (_, _) = mnist.load_data()
        # # Rescale -1 to 1
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)
        # y_train = y_train.reshape(-1, 1)

        # valid = np.ones((batch_size, 1))
        # fake = -np.ones((batch_size, 1))
        # #gene=self.data_loader.load_batch(batch_size)
        # for epoch in range(self.ite,epochs):
        #     # ---------------------
        #     #  Train Discriminator
        #     # ---------------------
        #     # Select a random half of images
        #     #imgs=next(gene)
        #     idx = np.random.randint(0, X_train.shape[0], batch_size)
        #     imgs = X_train[idx]
        #     img_labels=to_categorical(y_train[idx],self.classes)
        #     sampled_noise, sampled_labels = self.sample_generator_input(batch_size)
        #     gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)
        #     #noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        #     # Train the discriminator (real classified as ones and generated as zeros)
        #     d_loss = self.discriminator.train_on_batch([imgs, gen_input],
        #                                         [valid, fake,img_labels])

        #     # ---------------------
        #     #  Train Generator
        #     # ---------------------
        #     # Train the generator (wants discriminator to mistake images as real)
        #     g_loss = self.combined.train_on_batch(gen_input, [fake, sampled_labels])
        #     # Plot the progress
        #     print ("%d [D loss: %.2f, c_loss: %.2f] [g loss: %.2f c_g_loss: %.2f]" % (epoch, d_loss[1]+d_loss[2], d_loss[3], g_loss[1],g_loss[2]))
        #     #print(d_loss)
        #     #print(g_loss)
        #     self.d_loss.append(d_loss[0])
        #     self.g_loss.append(g_loss[1])
        #     self.c_loss.append(g_loss[2])
        #     #print(d_loss,g_loss)
        #     # If at save interval => save generated image samples
        #     if epoch % save_interval == 0:
        #         self.save_imgs(epoch)
        #     if epoch % (save_interval*10) == 0:
        #         self.save_model()
        #         self.save_log()
        #         self.plot_log()
        return 0
if __name__ == '__main__':
    myrcnn = MyRcnn()
    myrcnn.train(3000)
    
