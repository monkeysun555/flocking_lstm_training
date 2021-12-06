
# Tile setting:
# Matlab dir
tile_map_dir = '../tile_map/fov_map.mat'
new_tile_map_dir = '../tile_map/fov_map_90.mat'

## FoV dataset
num_video = 56
num_user = 31
tsinghua_fov_data_path = '../../Formated_Data/Experiment_2/'
tsinghua_pickled_data_path = '../pickled_data/'
tsinghua_seg_ave_attention_path = '../pickled_data/all_seg_attention.p'
tsinghua_seg_ave_attention_path_new = '../pickled_data/all_seg_attention_new.p'


shanghai_pickled_data_path = '../new_pickled_data/'
shanghai_seg_ave_attention_path = '../new_pickled_data/all_seg_attention.p'
shanghai_seg_ave_attention_path_new = '../new_pickled_data/all_seg_attention_new.p'
shanghai_seg_scanpath_path = '../new_pickled_data/all_seg_scanpath.p'
# Fov training
batch_size = 100
stride = 1
running_length = 10		# use past 10 seconds
predict_step = 5		# predict future 5 seconds 
data_chunk_stride = 1	
num_row = 16
num_col = 32
# num_row = 5
# num_col = 6
latent_dim = 16			#
new_latent_dim = 16
conv_kernel_size = 4
dropout_rate = 0.2
stateful_across_batch = False
model_saving_path = './models/'



predict_step_eva = 1