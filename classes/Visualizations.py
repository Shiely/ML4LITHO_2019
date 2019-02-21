
from tensorboard.plugins.beholder import Beholder
from tensorboard.plugins.beholder import BeholderHook

from tensorflow.contrib.tensorboard.plugins import projector
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from classes.SessionHook import SessionHook

import logging
import os
path_for_resnist_sprites = os.path.join('.', "resnist_images.png")

def prepare_projector(PROJECTORDIR, scopes):
	with tf.Session() as sess:
		config = projector.ProjectorConfig()
		print(scopes, enumerate(scopes))
		for i, scope in enumerate(scopes[:-1]): #the last scope is for labels
			print(i, scope)
			values=np.load(os.path.join(PROJECTORDIR, 'layer_activations_'+str(i)+'.npy'))
			embedding_var = tf.Variable(np.array(values), name='layer_activations_'+str(i))
			sess.run(embedding_var.initializer)
			embedding = config.embeddings.add()
			embedding.tensor_name = embedding_var.name


# Add metadata to the log
			embedding.metadata_path =  os.path.join('.', "metadata.tsv")
			print(embedding.metadata_path)
			writer = tf.summary.FileWriter(PROJECTORDIR, sess.graph)
			embedding.sprite.image_path = path_for_resnist_sprites
			embedding.sprite.single_image_dim.extend([21,21])
			projector.visualize_embeddings(writer, config)
			saver = tf.train.Saver([embedding_var])
		saver.save(sess, os.path.join(PROJECTORDIR, "model_emb.ckpt"), 1)
   
def create_sprite_image(images):
	"""Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
	if isinstance(images, list):
		images = np.array(images)
	img_h = images.shape[1]//2
	img_w = images.shape[2]//2
	n_plots = int(np.ceil(np.sqrt(images.shape[0])))
	spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))
	for i in range(n_plots):
		for j in range(n_plots):
			this_filter = i * n_plots + j
			if this_filter < images.shape[0]:
				this_img = images[this_filter][::2,::2]
				spriteimage[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w] = this_img #[::2,::2]
	return spriteimage

def vector_to_matrix_resnist(images):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(images,(-1,images.shape[1],images.shape[2]))

def invert_grayscale(images):
	""" Makes black white, and white black """
	return 1-images

def prepare_sprites(directory, images):
	to_visualize = images
	to_visualize = vector_to_matrix_resnist(to_visualize)
	to_visualize = invert_grayscale(to_visualize)

	sprite_image = create_sprite_image(to_visualize)

	plt.imsave(directory + path_for_resnist_sprites, sprite_image, cmap='gray')
	plt.imshow(sprite_image,cmap='jet')    
