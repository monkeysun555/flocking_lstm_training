import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Add, Softmax,Reshape
from keras.layers import Lambda,Concatenate,Flatten,ConvLSTM2D
from keras.layers import Permute,Conv2D
from keras import backend as K
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
# from tensorflow.keras.metrics import KLDivergence
import keras.losses
import os

import config as cfg
import pickle as pk
import numpy as np
import data_generator as dg
from utilities import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# import tensorflow as tf
# print("Num GPUs Available: ", len(tensorflow.config.experimental.list_physical_devices('GPU')))

# Utilities
# Concatenatelayer = Concatenate(axis=2)
Concatenatelayer1 = Concatenate(axis=-1)
expand_dim_layer = Lambda(lambda x: K.expand_dims(x,1))
# expand_dim_layer1 = Lambda(lambda x: K.expand_dims(x,axis=1))
# get_dim_layer = Lambda(lambda x: x[:,0,0,:,:])
get_dim_layer1 = Lambda(lambda x: x[:,0,:,:,:])
# flatten_layer = Flatten()

# scale_layer = Lambda(lambda x:x/K.sum(x,axis=(1,2)))
# scale_layer1 = Lambda(lambda x:x/K.sum(x,axis=(1)))
scale_layer2 = Lambda(lambda x:x/K.sum(x,axis=(1,2,3)))
# configuration
kernel_size = cfg.conv_kernel_size
latent_dim = cfg.latent_dim
row = cfg.num_row
col = cfg.num_col
epochs = 1000

input_shape1 = (cfg.running_length,row,col,1)           # Sample, time, row, col, channel
input_shape2 = (cfg.running_length,row,col,latent_dim*2)
input_shape3 = (cfg.running_length,row,col,latent_dim)


# convLSTM for target past segment average
encoder_inputs = Input(shape=(cfg.running_length, row, col, 1))
convlstm_encoder = ConvLSTM2D(filters=latent_dim*2, kernel_size=(kernel_size, kernel_size),
                   input_shape=input_shape1, dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                   stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                   padding='same', return_sequences=True, return_state=True)
pst_outputs_sqns, pst_state_h0, pst_state_c0 = convlstm_encoder(encoder_inputs)
states0 = [pst_state_h0, pst_state_c0]
# print(convlstm_encoder)

convlstm_encoder1 = ConvLSTM2D(filters=latent_dim, kernel_size=(kernel_size, kernel_size),
                   input_shape=input_shape2, dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                   stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                   padding='same', return_sequences=True, return_state=True)
pst_outputs_sqns, pst_state_h1, pst_state_c1 = convlstm_encoder1(pst_outputs_sqns)
states1 = [pst_state_h1, pst_state_c1]

convlstm_encoder2 = ConvLSTM2D(filters=latent_dim//2, kernel_size=(kernel_size, kernel_size),
                   input_shape=input_shape3, dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                    stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                    padding='same', return_sequences=True, return_state=True)
# print(pst_outputs_sqns.shape())
pst_outputs_sqns, pst_state_h2, pst_state_c2 = convlstm_encoder2(pst_outputs_sqns)
states2 = [pst_state_h2, pst_state_c2]

# print(pst_outputs_sqns)

# ###======convLSTM on target future decoder======
decoder_inputs = Input(shape=(1,row,col,1))   # Only last sequence from encoder
convlstm_decoder = ConvLSTM2D(filters=latent_dim*2, kernel_size=(kernel_size, kernel_size),
                   input_shape=input_shape1,dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                   stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                   padding='same', return_sequences=True, return_state=True)

convlstm_decoder1 = ConvLSTM2D(filters=latent_dim, kernel_size=(kernel_size, kernel_size),
                   input_shape=input_shape2,dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                   stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                   padding='same', return_sequences=True, return_state=True)

convlstm_decoder2 = ConvLSTM2D(filters=latent_dim//2, kernel_size=(kernel_size, kernel_size),
                   input_shape=input_shape3,dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                   stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                   padding='same', return_sequences=True, return_state=True)


# ### 2D conv
pred_conv_conv = Conv2D(filters=32, kernel_size=(kernel_size,kernel_size), padding='same',
    activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
pred_conv_conv1 = Conv2D(filters=64, kernel_size=(kernel_size,kernel_size), padding='same',
    activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
pred_conv_conv2 = Conv2D(filters=1, kernel_size=(kernel_size,kernel_size), padding='same',   # filters=fps
    activation='relu', use_bias=True, kernel_initializer='glorot_uniform')


pred_interval_conv1 = Conv2D(filters=4, kernel_size=(kernel_size,kernel_size), padding='same',
    activation='relu', use_bias=True, kernel_initializer='glorot_uniform')

# 2D conv for other users' gt
# others_conv0 = Conv2D(filters=16, kernel_size=(kernel_size,kernel_size), padding='same',
#     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
# others_conv1 = Conv2D(filters=32, kernel_size=(kernel_size,kernel_size), padding='same',
#     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
# # others_conv2 = Conv2D(filters=1, kernel_size=(kernel_size,kernel_size), padding='same',   # filters=fps
# #     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')

# # 2D conv for other users' var gt
# others_var_conv0 = Conv2D(filters=16, kernel_size=(kernel_size,kernel_size), padding='same',
#     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
# others_var_conv1 = Conv2D(filters=32, kernel_size=(kernel_size,kernel_size), padding='same',
#     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
# others_var_conv2 = Conv2D(filters=1, kernel_size=(kernel_size,kernel_size), padding='same',   # filters=fps
#     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')

# Dense for number of users
# other_user_dense = Dense(8, input_shape=(1,),activation='relu')

# output final map for next timestamp
# final_output = Dense(row*col, input_shape=(2*row*col+16,),activation='softmax')
# final_reshape = Reshape((row,col,1), input_shape=(row*col,))

# bnlayer0 = BatchNormalization(axis=-1,center=True,scale=True)
# bnlayer1 = BatchNormalization(axis=-1,center=True,scale=True)
# bnlayer2 = BatchNormalization(axis=-1,center=True,scale=True)

# 
all_outputs= []
inputs = decoder_inputs
# other_inputs = Input(shape=(cfg.predict_step, row, col, 1))
# other_inputs_var = Input(shape=(cfg.predict_step, row, col, 1))
pred_inteval = Input(shape=(cfg.predict_step, row, col, 1))

# num_users = Input(shape=(1))
for time_ind in range(cfg.predict_step):
    # print('k0', inputs.shape)
    fut_outputs_sqns0, fut_state_h, fut_state_c = convlstm_decoder([inputs]+states0)
    states0 = [fut_state_h, fut_state_c]
    fut_outputs_sqns1, fut_state_h, fut_state_c = convlstm_decoder1([fut_outputs_sqns0]+states1)
    states1 = [fut_state_h, fut_state_c]
    fut_outputs_sqns2, fut_state_h, fut_state_c = convlstm_decoder2([fut_outputs_sqns1]+states2)
    states2 = [fut_state_h, fut_state_c]

    fut_outputs_sqns = Concatenatelayer1([fut_outputs_sqns0,fut_outputs_sqns1,fut_outputs_sqns2])
    # print(fut_outputs_sqns.shape)
    
    ## For coolborative prediction
    # Concatenate ground truth and var
    # groud_truth_map = other_inputs[:,time_ind,:,:,:]
    # # print(groud_truth_map.shape)
    # groud_truth_map_var = other_inputs_var[:,time_ind,:,:,:]

    # groud_truth_map = others_conv0(groud_truth_map)
    # groud_truth_map = others_conv1(groud_truth_map)
    pred_interval_gt = pred_inteval[:,time_ind,:,:,:]

    pre_interval_map = pred_interval_conv1(pred_interval_gt)
    pre_interval_map = expand_dim_layer(pre_interval_map)

    # groud_truth_map_var = others_var_conv0(groud_truth_map_var)
    # groud_truth_map_var = others_var_conv1(groud_truth_map_var)    

    # # print(groud_truth_map_var.shape)

    # groud_truth_map = expand_dim_layer(groud_truth_map)
    # groud_truth_map_var = expand_dim_layer(groud_truth_map_var)

    # fut_outputs_sqns = Concatenatelayer1([fut_outputs_sqns, groud_truth_map, groud_truth_map_var])

    fut_outputs_sqns = Concatenatelayer1([fut_outputs_sqns, pre_interval_map])

    ### predict others' future
    outputs = get_dim_layer1(fut_outputs_sqns)
    # print('k1', outputs.shape)
    outputs = pred_conv_conv(outputs)
    # print('k1', outputs.shape)
    # outputs = bnlayer0(outputs)
    outputs = pred_conv_conv1(outputs)
    # outputs = bnlayer1(outputs)
    outputs = pred_conv_conv2(outputs)

    # print('f3', outputs.shape)
    outputs = scale_layer2(outputs)
    outputs = expand_dim_layer(outputs)
    # print('final', outputs.shape)
    inputs = outputs

    all_outputs.append(outputs)

# Concatenate all predictions
# print('all_outputs', all_outputs)
decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
final_shape = Reshape((cfg.predict_step,row*col), input_shape=(cfg.predict_step,row,col,1))
decoder_outputs = final_shape(decoder_outputs)

print('encoder_input', encoder_inputs.shape)
print('decoder_input:', decoder_inputs.shape)
# print('other_inputs: ', other_inputs.shape)
# print('other_inputs_var: ', other_inputs_var.shape)

model = Model([encoder_inputs,decoder_inputs, pred_inteval],decoder_outputs)
# return 
# RMSprop = optimizers.RMSprop(lr=0.01,clipnorm=3)
# sgd = optimizers.sgd(lr=0.0001,clipnorm=1)
# weights = np.ones(30)
KL = tf.keras.losses.KLDivergence()
model.compile(loss=KL, optimizer='adam')
# model.compile(loss='mse', optimizer='adam', metrics= [sphere_loss])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[KL])#costfunc.likelihood_loss())#'categorical_crossentropy')



datadb = pk.load(open(cfg.shanghai_seg_ave_attention_path,'rb'))        # shape: (9, 48, seconds, row, column)
num_samples = 100
mygenerator = dg.generator_train_tar(datadb)
mygenerator_val = dg.generator_train_tar(datadb, phase='val')

## Training
model_path = cfg.model_saving_path
file_path = model_path + 'non_heatmap/'
if not os.path.exists(file_path):
    os.mkdir(file_path)
    
# for filename in os.listdir(file_path):
#     file_path = os.path.join(file_path, filename)
#     try:
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#     except Exception as e:
#         print('Failed to delete %s. Reason: %s' % (file_path, e))

tag = 'convLSTMtar_seqseq_shanghai'
model_checkpoint = ModelCheckpoint(file_path+tag+'_epoch{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_weights_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=5, min_lr=1e-5)
stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
# # model.fit([encoder_input_data, decoder_input_data],decoder_target_data,
# #          batch_size=batch_size, 
# #          epochs=epochs,
# #          validation_split=0.2,
# #          shuffle=cfg.shuffle_data, initial_epoch=0,
# #          callbacks=[model_checkpoint, reduce_lr, stopping])

# Save arch
with open('./arch/non_heatmap/model_architecture.json', 'w') as f:
    f.write(model.to_json())
    print("model arch saved")
model.fit(mygenerator, steps_per_epoch=num_samples,epochs=epochs,
                  batch_size=cfg.batch_size, 
                  validation_data=mygenerator_val,validation_steps=100,
                  callbacks=[model_checkpoint,reduce_lr],
                  use_multiprocessing=False)

### ====================Testing====================
# mygenerator_test=generator_train2(datadb,phase='test')
# model.load_weights('convLSTMtar_seqseq_tsinghua_epoch33-0.1125.h5')


