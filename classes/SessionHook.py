import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from tensorflow.contrib.tensorboard.plugins import projector

class SessionHook(tf.train.SessionRunHook):

    def __init__(self, path, scopes):
        super(SessionHook, self).__init__()

        self.iterator_initializer_func = None

#        self._tensors = None

#        values = scopes['net_scope'] + '/' + scopes['emb_scope'] + '/' + scopes['emb_name'] + '/' + 'Relu:0'
#        labels = scopes['net_scope'] + '/' + scopes['metrics_scope'] + '/' + scopes['label_name'] + ':0'
        self._tensors = None

        self._tensor_names = scopes
        self._path = path
        #print('hook __init__', scopes, self._tensor_names)
        self._embeddings = [list([]) for _ in range(len(scopes))]   

    def begin(self):
        self._tensors = [tf.get_default_graph().get_tensor_by_name(x) for x in self._tensor_names]
        #print('hook begin', self._tensors)

    def after_create_session(self, session, coord):
        """ Initialise the iterator after the session has been created."""
        #print('hook after_create_session', session, coord)
        #self.iterator_initializer_func(session)

    def before_run(self, run_context):
        #print('hook before run') #, self._tensors)
        return tf.train.SessionRunArgs(self._tensors)

    def after_run(self, run_context, run_values):
        #print('hook after_run')
        #print(len(run_values.results))
        [self._embeddings[x].extend(run_values[0][x]) for x in range(len(run_values.results))]
#        self._embeddings[0].extend(run_values[0][0])
#        self._embeddings[1].extend(run_values[0][1])

    def end(self, session):
        print('hook end')
        classes = ['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1']
        embeddings = self.get_embeddings()

        
        values = embeddings['values']
        labels = np.array(embeddings['labels']).ravel()
        captions = [classes[x] for x in labels]

        if not os.path.isdir(self._path):
            os.makedirs(self._path)

        with open(os.path.join(self._path, 'metadata.tsv'), 'w+') as f:
            f.write('Index\tCaption\tLabel\n')
            for idx in range(len(labels)):
                f.write('{:05d}\t{}\t{}\n'
                .format(idx, captions[idx], labels[idx]))
            f.close()

        i=0
        pca=PCA(n_components=10)
        for value in values:
            print(len(value[0]))
            feat_cols = ['pixel'+str(i) for i in range(len(value[0]))]
            df = pd.DataFrame(value, columns=feat_cols)
            pca_result=pca.fit_transform(df[feat_cols].values)
            print('Cumulative explained variation for 10 principal components: {} layer {}'.format(np.sum(pca.explained_variance_ratio_),i))
            np.save(os.path.join(self._path, "layer_activations_" + str(i) + ".npy"),pca_result)
            i+=1
       
        # The embedding variable to be stored


    def get_embeddings(self):
        #print('get embeddings') #, len(self._embeddings), self._embeddings[0][:100],self._embeddings[-1][:100])
        return {
            'values': self._embeddings[0:-1],
            'labels': self._embeddings[-1],
        }

    def set_iterator_initializer(self, fun):
        #print('hook iterator initializer',fun)
        self.iterator_initializer_func = fun
