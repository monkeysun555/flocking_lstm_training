import config as cfg
import numpy as np

def get_data_per_vid_per_target(per_video_db,_video_ind,_target_user,num_user=cfg.num_user,stride=cfg.data_chunk_stride):
    per_video_db_tar = per_video_db[_target_user,:][np.newaxis,:,:]
    per_video_db_oth = np.delete(per_video_db,_target_user,axis=0)
    # print('shape of per_video_db_tar is: ',per_video_db_tar.shape)
    # print('shape of per_video_db_oth is: ',per_video_db_oth.shape)

    num_oth = np.random.randint(1, num_user-1)
    oth_user_id = np.random.choice(num_user-1, num_oth, replace=False)
    print("sampled users: ", len(oth_user_id))
    per_video_db_tar, per_video_db_future_tar, per_video_db_future_input_tar,other_heat_map,other_heat_map_var,num_oth, pred_interval = get_past_future(per_video_db_tar, per_video_db_oth, oth_user_id,num_oth)
    
    # print('_video_ind = ',_video_ind,'per_video_db.shape = ',per_video_db.shape)  
    # print('_video_ind = ',_video_ind,'per_video_db_tar.shape = ',per_video_db_tar.shape) 
    # print('_video_ind = ',_video_ind,'per_video_db_future_tar.shape = ',per_video_db_future_tar.shape) 
    # print('_video_ind = ',_video_ind,'per_video_db_future_input_tar.shape = ',per_video_db_future_input_tar.shape) 
    # print('_video_ind = ',_video_ind,'other_heat_map.shape = ',other_heat_map.shape)
    # print('_video_ind = ',_video_ind,'num_oth.shape = ',num_oth.shape)

    return per_video_db_tar, per_video_db_future_tar, per_video_db_future_input_tar,other_heat_map,other_heat_map_var,num_oth, pred_interval

def generator_train(datadb,phase='train'):   #datadb
    max_encoder_seq_length = cfg.running_length
    batch_size = 1                                  #cfg.batch_size
    stride=cfg.data_chunk_stride
    # print('entered')
    if phase not in ['val','test']:
        # stride=10     #test stride=10
        data_keys = [0,5,8]
    else:
        # data_keys = [1,2,4,3,6]
        data_keys = [1,3,6]

    # print(len(datadb[0]))
    # for j in range(len(datadb[0])):
    #   print(len(datadb[0,j]))
    while True:
        ii= np.random.randint(0, len(data_keys)-1)
        _video_ind = data_keys[ii%len(data_keys)]
        print('video id', _video_ind)
        per_video_db = np.stack(datadb[_video_ind,:])
        # per_video_db.reshape(per_video_db.shape[0], len(per_video_db[0]), cfg.num_row, cfg.num_col)
        # per_video_db = util.cut_head_or_tail_less_than_1sec(per_video_db)         # From chenge
        # print('shape of per_video_db is: ',per_video_db.shape)  
        # per_video_db = np.reshape(per_video_db,(per_video_db.shape[0], per_video_db.shape[1], 16, 32))        #Liyang shape: (#users, #second, #rows, #cols)
        # print('shape of per_video_db is: ',per_video_db.shape)
        assert per_video_db.shape[1] >= cfg.running_length*2
        # print(per_video_db.shape[0])
        for _target_user in range(per_video_db.shape[0]):
            print('target user:',_target_user)
            # if _target_user == 5:
            #     continue
            # print(_target_user)
            encoder_input_data,decoder_target_data,decoder_input_data,other_heat_map,other_heat_map_var,num_oth,pred_interval = get_data_per_vid_per_target(per_video_db,_video_ind,_target_user,stride=stride)

            local_counter=0
            while local_counter<encoder_input_data.shape[0]-10:
                print(local_counter,local_counter+batch_size)
                # print('encoder input: ', encoder_input_data[local_counter:local_counter+batch_size,:].shape)
                # # print('decoder input: ', decoder_target_data[local_counter:local_counter+batch_size,:].shape)
                # # print('decoder output: ', decoder_target_data[local_counter:local_counter+batch_size,0,:,:,:])
                # print('average: ', other_heat_map[local_counter:local_counter+batch_size,:])
                # print('variance ', other_heat_map_var[local_counter:local_counter+batch_size,:])
                # print('decoder target: ', decoder_target_data[local_counter:local_counter+batch_size,:])
                # b = decoder_target_data[local_counter:local_counter+batch_size,:]
                # print(b.shape)
                # reshape y_true
                # print()
                yield [encoder_input_data[local_counter:local_counter+batch_size,:,:,:],\
                          #others_fut_input_data[local_counter:local_counter+batch_size,:],\
                          decoder_input_data[local_counter:local_counter+batch_size,:],\
                          other_heat_map[local_counter:local_counter+batch_size,:],\
                          other_heat_map_var[local_counter:local_counter+batch_size,:],\
                          num_oth[local_counter:local_counter+batch_size,:],\
                          pred_interval[local_counter:local_counter+batch_size,:]],\
                          decoder_target_data[local_counter:local_counter+batch_size,:]
                          # decoder_target_data[local_counter:local_counter+batch_size,:]
                # print("sample yield, shape is: ", encoder_input_data[local_counter:local_counter+batch_size,:].shape)  
                # print("number of random users: ", num_oth)                  
                # print("encoder input: ", encoder_input_data)
                # print("decoder_input:", )
                local_counter+=batch_size

        # ii+=1#next video

# def generator_train_tar(datadb,phase='train'):   #datadb
#     max_encoder_seq_length = cfg.running_length
#     batch_size = 1                                  #cfg.batch_size
#     stride=cfg.data_chunk_stride
#     # print('entered')
#     if phase in ['val','test']:
#         # stride=10     #test stride=10
#         data_keys = [2,6]
#     else:
#         data_keys = [0,1,3,4,5,7,8]
#     # print(len(datadb[0]))
#     # for j in range(len(datadb[0])):
#     #   print(len(datadb[0,j]))
#     ii=0
#     while True:
#         _video_ind = data_keys[ii%len(data_keys)]
#         per_video_db = np.stack(datadb[_video_ind,:])
#         # per_video_db.reshape(per_video_db.shape[0], len(per_video_db[0]), cfg.num_row, cfg.num_col)
#         # per_video_db = util.cut_head_or_tail_less_than_1sec(per_video_db)         # From chenge
#         # print('shape of per_video_db is: ',per_video_db.shape)  
#         # per_video_db = np.reshape(per_video_db,(per_video_db.shape[0], per_video_db.shape[1], 16, 32))        #Liyang shape: (#users, #second, #rows, #cols)
#         # print('shape of per_video_db is: ',per_video_db.shape)
#         assert per_video_db.shape[1] >= cfg.running_length*2
#         # print(per_video_db.shape[0])
#         for _target_user in range(per_video_db.shape[0]):
#             # print('target user:',_target_user)
#             # print(_target_user)
#             encoder_input_data,decoder_target_data,decoder_input_data,other_heat_map,num_oth = get_data_per_vid_per_target(per_video_db,_video_ind,_target_user,stride=stride)

#             local_counter=0
#             while local_counter<encoder_input_data.shape[0]:
#                 # print(local_counter,local_counter+batch_size)
#                 # print(encoder_input_data[local_counter:local_counter+batch_size,:].shape)
#                 # print("before yield ", decoder_target_data[local_counter:local_counter+batch_size,:].shape)
#                 # print(decoder_target_data[local_counter:local_counter+batch_size,0,:,:,:])
#                 # b = decoder_target_data[local_counter:local_counter+batch_size,:]
#                 # print(b.shape)
#                 # reshape y_true
#                 yield [encoder_input_data[local_counter:local_counter+batch_size,:,:,:],\
#                           decoder_input_data[local_counter:local_counter+batch_size,:]],\
#                           decoder_target_data[local_counter:local_counter+batch_size,:]
#                           # decoder_target_data[local_counter:local_counter+batch_size,:]
#                 # print("sample yield, shape is: ", encoder_input_data[local_counter:local_counter+batch_size,:].shape)  
#                 # print("number of random users: ", num_oth)                  
#                 # print("encoder input: ", encoder_input_data)
#                 # print("decoder_input:", )
#                 local_counter+=batch_size

#         ii+=1#next video

def get_past_future(per_video_db, oth_user_video_db, oth_user_id, num_oth, stride=cfg.data_chunk_stride):
    # print(per_video_db.shape)
    per_video_db = np.expand_dims(per_video_db, axis=4)
    oth_user_video_db = np.expand_dims(oth_user_video_db[oth_user_id, :,:,:], axis=4)
    oth_user_video_var = np.std(oth_user_video_db, axis=0, keepdims=True)   ## MUST BEFORE GET MEAN, AS MEAN CHANGE ORIGINAL VALUES
    oth_user_video_db = np.mean(oth_user_video_db, axis=0, keepdims=True)
    # print('var all: ', oth_user_video_var)
    # print(oth_user_video_db.shape)
    max_encoder_seq_length = cfg.running_length
    pre_seq_length = cfg.predict_step
    assert per_video_db.shape[1]>=max_encoder_seq_length*2
    # print(per_video_db.shape)
    num_batch = (per_video_db.shape[1]-pre_seq_length)//stride
    encoder_input = np.zeros((num_batch,max_encoder_seq_length,per_video_db.shape[2],per_video_db.shape[3],1))
    # encoder_future = np.zeros((num_batch,pre_seq_length,per_video_db.shape[2],per_video_db.shape[3],1))
    encoder_future = np.zeros((num_batch,pre_seq_length,per_video_db.shape[2]*per_video_db.shape[3]))
    decoder_input = np.zeros((num_batch,1,per_video_db.shape[2],per_video_db.shape[3],1))
    other_gt =  np.zeros((num_batch,pre_seq_length,per_video_db.shape[2],per_video_db.shape[3],1))
    other_var_gt =  np.zeros((num_batch,pre_seq_length,per_video_db.shape[2],per_video_db.shape[3],1))
    num_other_gt =  np.ones((num_batch,pre_seq_length,per_video_db.shape[2],per_video_db.shape[3],1))*num_oth/48/per_video_db.shape[2]/per_video_db.shape[3]
    # print(num_other_gt.shape, num_other_gt)
    pred_interval_gt = np.zeros((num_batch,pre_seq_length,per_video_db.shape[2],per_video_db.shape[3],1))
    #
    pred_interval = np.ones((pre_seq_length,per_video_db.shape[2],per_video_db.shape[3],1))
    for i in range(pred_interval.shape[0]):
        pred_interval[i] *= (i+1)/pred_interval.shape[0]/per_video_db.shape[2]/per_video_db.shape[3]
    print("This is pred_interval: ", pred_interval)
    # prepare encoding/decoding data
    past = np.zeros((max_encoder_seq_length,per_video_db.shape[2],per_video_db.shape[3],1))
    future = per_video_db[0,:pre_seq_length,:,:,:]
    # print(future.shape)
    # Get average of others
    other = oth_user_video_db[0,:pre_seq_length,:,:,:]
    other_var = oth_user_video_var[0,:pre_seq_length,:,:,:]
    # pred_interval = pred_interval_gt[0,:pre_seq_length,:,:,:]

    # print(other.shape)
    for i in range(per_video_db.shape[1]-pre_seq_length):
        past = np.roll(past,-1,axis=0)
        future = np.roll(future,-1,axis=0)
        other = np.roll(other,-1,axis=0)
        other_var = np.roll(other_var,-1,axis=0)
        # pred_interval = np.roll(other_var,-1,axis=0)
        # print(oth_user_video_db[0,i+pre_seq_length,:,:,:].shape)
        past[-1,:,:,:] = per_video_db[0,i,:,:,:]
        future[-1,:,:,:] = per_video_db[0,i+pre_seq_length,:,:,:]
        other[-1,:,:,:] = oth_user_video_db[0,i+pre_seq_length,:,:,:]
        other_var[-1,:,:,:] = oth_user_video_var[0,i+pre_seq_length,:,:,:]
        # Add to seq
        encoder_input[i,:,:,:,:] = past
        # encoder_future[i,:,:,:,:] = future
        encoder_future[i,:,:] = future.reshape((pre_seq_length,cfg.num_row*cfg.num_col))
        decoder_input[i,:,:,:,:] = past[-1,:,:,:]
        other_gt[i,:,:,:,:] = other
        other_var_gt[i,:,:,:,:] = other_var
        pred_interval_gt[i,:,:,:,:] = pred_interval
    print(other_var_gt.shape, num_other_gt.shape, pred_interval_gt.shape)
    return encoder_input,encoder_future,decoder_input,other_gt,other_var_gt,num_other_gt,pred_interval_gt
