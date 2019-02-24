import numpy as np

def loadResNIST_legacy(DATADIR, patch_size):

	min_patch_index=49-patch_size
	max_patch_index=patch_size+1
	patch_center_index=patch_size//2
	images = np.load(DATADIR+'./ai_images_train.npy') # Returns np.array
	train_data = np.asarray([image[min_patch_index:max_patch_index,min_patch_index:max_patch_index] for image in list(images)])
	resist = np.load(DATADIR+'./res_contours_train.npy')
	train_labels = np.asarray([np.mean(image[patch_center_index:patch_center_index+1,patch_center_index:patch_center_index+1])*10.0 for image in list(resist)], dtype=np.int32)

	images= np.load(DATADIR+'./ai_images_test.npy')
	eval_data = np.asarray([image[min_patch_index:max_patch_index,min_patch_index:max_patch_index] for image in list(images)])
	resist = np.load(DATADIR+'./res_contours_test.npy')
	eval_labels = np.asarray([np.mean(image[patch_center_index:patch_center_index+1,patch_center_index:patch_center_index+1])*10.0 for image in list(resist)], dtype=np.int32)
	resist=[]
	images=[]
	return train_data, train_labels, eval_data, eval_labels

	
def loadResNIST(DATADIR, image_size, train_images, eval_images):
	import zipfile
	import numpy as np
	datafiles = ['train_images', 'train_labels', 'eval_images', 'eval_labels']
	for file in datafiles:
		file_ref = zipfile.ZipFile( DATADIR + file + '.zip', 'r' )
		file_ref.extractall('./resNIST')
		file_ref.close()
	train_data = np.load(DATADIR+'/balanced_train_data.npy')
	train_labels = np.load(DATADIR+'/balanced_train_labels.npy')
	eval_data = np.load(DATADIR+'/balanced_eval_data.npy') 
	eval_labels = np.load(DATADIR+'/balanced_eval_labels.npy') 

	return train_data[:train_images], train_labels[:train_images], eval_data[:eval_images], eval_labels[:eval_images]