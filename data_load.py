import config as cfg
import csv
import numpy as np
import pickle as pk
from scipy.io import loadmat
import os

def get_fps():
    data_dir = cfg.shanghai_fov_data_path + 'videoMeta.csv'
    infos = []
    with open(data_dir, newline='') as csvfile:
        vidoe_info = csv.reader(csvfile)
        next(vidoe_info)
        for row in vidoe_info:
            infos.append([int(row[0]), row[2], int(row[3])])
    # print(infos)
    return infos


# def heat_map_shanghai_data(fdir=cfg.shanghai_fov_data_path):
#     for v_id in range(cfg.num_video):
#         for u_id in range(cfg.num_user):
#             fov_data_path = fdir + str(v_id) + '/' + str(u_id) + '.p'
#             pickle_data = pk.load(open(fov_data_path, "rb"))
#             user_y, user_p, user_r = get_ypr_from_shanghai_data(pickle_data)

# Get segment average attention
# Using Yixiang's map
def get_attention_from_frame_ypr(curr_segment_frames, tile_map):
    if not len(curr_segment_frames): return []
    attention = np.zeros((cfg.num_row, cfg.num_col))
    # n_frames = len(curr_segment_frames)
    # frame_ratio = 1./len(curr_segment_frames)
    for frame in curr_segment_frames:
        # Get degree (pitch, yaw)
        # print(frame)
        # print(frame[0], frame[1])
        pitch_in_degree = int(frame[1][1]/np.pi*180)+90
        yaw_in_degree = int(frame[1][0]/np.pi*180)+180
        # print(pitch_in_degree, yaw_in_degree)
        frame_map = tile_map[pitch_in_degree][yaw_in_degree]
        frame_map = np.dot(frame_map, 1./np.sum(frame_map))
        attention += frame_map/len(curr_segment_frames)
    # attention = np.dot(attention, 1./np.sum(attention))
    # print(attention)
    assert np.round(np.sum(attention),2) == 1
    return attention

def translate_shanghai_to_attention(fdir=cfg.shanghai_pickled_data_path):
    # Load mapping
    # ypr_to_tile = np.array(loadmat(cfg.tile_map_dir)['map'])
    # print(ypr_to_tile[0]) # 181*361*row*col
    ypr_to_tile = np.array(loadmat(cfg.new_tile_map_dir)['map'])

    # print(ypr_to_tile[0,0,:,:]) # 181*361*row*col
    # return

    all_data = []       # reshape later
    for v_id in range(300):
        video_attention = []
        for u_id in range(31):
            print("Video %d, user %d"%(v_id, u_id))
            user_attention = []
            # print(fdir + str(v_id))
            if not os.path.isdir(fdir + str(v_id)):
                continue
            # print('enter')
            fov_data_path = fdir + str(v_id) + '/' + str(u_id) + '.p'
            pickle_data = pk.load(open(fov_data_path, "rb"))    # [[time, (y,p,r)],.....]
            # print(pickle_data)
            # user_y, user_p, user_r = get_ypr_from_shanghai_data(pickle_data)
            for seg_id in range(len(pickle_data)):
                time = pickle_data[seg_id][0]
                frame_ypr = pickle_data[seg_id][1:]
                    ## Get attention of previous segment
                    ## Translate (yaw, pitch) to attention 
                attention = get_attention_from_frame_ypr(frame_ypr, ypr_to_tile) # 16*32, segment average attention
                # print(attention)
                user_attention.append(attention)
            print(len(user_attention))
            video_attention.append(user_attention)
        if len(video_attention):
            all_data.append(video_attention)
    print(len(all_data))
    all_data = np.array(all_data)             # 9 * 48 * second * row * col
    for i in range(len(all_data)):
        print(len(all_data[i][0]), i)
    # print(len(all_data[0][0][0]))
    # Save segment average attention
    # pk.dump(all_data, open(cfg.shanghai_seg_ave_attention_path, "wb")) 
    # pk.dump(all_data, open(cfg.shanghai_seg_ave_attention_path_new, "wb")) 
    pk.dump(all_data, open(cfg.shanghai_seg_ave_attention_path, "wb")) 



def translate_shanghai_to_binary_path(fdir=cfg.shanghai_pickled_data_path):
    # Load mapping
    # ypr_to_tile = np.array(loadmat(cfg.tile_map_dir)['map'])
    # print(ypr_to_tile[0]) # 181*361*row*col
    # ypr_to_tile = np.array(loadmat(cfg.new_tile_map_dir)['map'])
    # print(ypr_to_tile[0,0,:,:]) # 181*361*row*col

    all_data = []       # reshape later
    for v_id in range(300):
        video_scanpath = []
        for u_id in range(31):
            print("Video %d, user %d"%(v_id, u_id))
            user_scanpath = []
            # print(fdir + str(v_id))
            if not os.path.isdir(fdir + str(v_id)):
                continue
            # print('enter')
            fov_data_path = fdir + str(v_id) + '/' + str(u_id) + '.p'
            pickle_data = pk.load(open(fov_data_path, "rb"))    # [[time, (y,p,r)],.....]
            
            for seg_id in range(len(pickle_data)):
                time = pickle_data[seg_id][0]
                frame_ypr = pickle_data[seg_id][1:]
                scanpath = get_scanpath_from_frames(frame_ypr)
                print(np.sum(scanpath))
                user_scanpath.append(scanpath)
            print(len(user_scanpath))
            video_scanpath.append(user_scanpath)
        if len(video_scanpath):
            all_data.append(video_scanpath)
    print(len(all_data))
    all_data = np.array(all_data)             # 9 * 48 * second * row * col
    for i in range(len(all_data)):
        print(len(all_data[i][0]), i)
    # print(len(all_data[0][0][0]))
    # Save segment average attention
    # pk.dump(all_data, open(cfg.shanghai_seg_ave_attention_path, "wb")) 
    # pk.dump(all_data, open(cfg.shanghai_seg_ave_attention_path_new, "wb")) 
    pk.dump(all_data, open(cfg.shanghai_seg_scanpath_path, "wb"))

def get_scanpath_from_frames(frames):
    binay_map = [[0 for _ in range(cfg.num_col)] for _ in range(cfg.num_row)]
    for frame in frames:
        pitch_in_degree = int(frame[1][1]/np.pi*180)+90
        yaw_in_degree = int((frame[1][0]/np.pi*180+180)%360)
        row_idx, col_idx = int(pitch_in_degree/180*cfg.num_row), int(yaw_in_degree/360*cfg.num_col)
        binay_map[row_idx][col_idx] += 1
    return binay_map


def get_ypr_from_shanghai_data(tmp):
    """for shanghai dataset"""
    """get euler angle list from the file lines"""
    roll_list = []
    pitch_list = []
    yaw_list = []
    # HmdPosition_x_list = []
    # HmdPosition_y_list = []
    # HmdPosition_z_list = []
    for ii in range(1,len(tmp)):
        [q1,q2,q3,q0] = np.array(tmp[ii].split(','))[2:6]
        #!!!order not one to one! https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        # q0,q1,q2,q3 = qw,qx,qy,qz 
        yaw, pitch, roll = quaternion2euler2(q0,q1,q2,q3)
        yaw_list.append(yaw)
        pitch_list.append(pitch)
        roll_list.append(roll)
        # [x,y,z] = np.array(tmp[ii].split(','))[6:]
        # HmdPosition_x_list.append(x)
        # HmdPosition_y_list.append(y)
        # HmdPosition_z_list.append(z)
    return yaw_list, pitch_list, roll_list

def quaternion2euler2(q):
    q0 = np.float(q[0])
    q1 = np.float(q[1])
    q2 = np.float(q[2])
    q3 = np.float(q[3])
    """convert quaternion tuple to euler angles"""
    roll = np.arctan2(2*(q0*q1+q2*q3),(1-2*(q1**2+q2**2)))
    # confine to [-1,1] to avoid nan from arcsin
    sintemp = min(1,2*(q0*q2-q3*q1))
    sintemp = max(-1,sintemp)
    pitch = np.arcsin(sintemp)
    yaw = np.arctan2(2*(q0*q3+q1*q2),(1-2*(q2**2+q3**2)))
    # assert np.abs(yaw) <= np.pi
    # assert np.abs(pitch) <= 0.5*np.pi
    return yaw, pitch, roll

def main():
    user_num = 48
    video_num = 9
    
    ### load shanghai data and translate to yaw_pitch_roll
    # data_dir = cfg.shanghai_fov_data_path
    translate_shanghai_to_attention(cfg.shanghai_pickled_data_path)
    translate_shanghai_to_binary_path(cfg.shanghai_pickled_data_path)

if __name__ == '__main__':
    main()
