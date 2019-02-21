import tensorflow as tf

from datasets.cifar10 import Cifar10

from classes.Estimator import Estimator

datasets = {
    'cifar-10': Cifar10
}


class Experiment:

    def __init__(self, params):

        # Define model parameters
        self.params = params

        # Set the run_config and the directory to save the model and stats
        self.run_config = tf.contrib.learn.RunConfig()

    def get_experiment_fn(self, run_config, params):

        """Create an experiment to train and evaluate the model.
        Args:
            run_config (RunConfig): Configuration for Estimator run.
            params (HParam): Hyperparameters
        Returns:
            (Experiment) Experiment for training the mnist model.
        """
        # You can change a subset of the run_config properties as
        # run_config = run_config.replace(save_checkpoints_steps=params.min_eval_frequency)

        # Define the mnist classifier
        estimator = Estimator(run_config, params).get_estimator()

        # emb_saver_hook = params.hooks['embeddings_hook']

        train_input_fn, train_input_hook = self.get_train_inputs_fn(params)
        eval_input_fn, eval_input_hook = self.get_test_inputs_fn(params)

        # Define the experiment
        return tf.contrib.learn.Experiment(
            estimator=estimator,  # Estimator
            train_input_fn=train_input_fn,  # First-class function
            eval_input_fn=eval_input_fn,  # First-class function
            train_steps=params.train_steps,  # Minibatch steps
            # min_eval_frequency=params.min_eval_frequency,  # Eval frequency
            eval_delay_secs=0,
            train_monitors=[train_input_hook],  # Hooks for training
            eval_hooks=[eval_input_hook],  # Hooks for evaluation
            eval_steps=None  # Use evaluation feeder until its empty
        )

    """ -------------------------------------------- Datasets provider -------------------------------------------- """

    @staticmethod
    def get_dataset_provider(params):
        if params.model not in datasets:
            raise ValueError('Name of dataset unknown %s' % params.model)
        return datasets[params.model](params.dataset_dir)

    @staticmethod
    def get_test_inputs_fn(params):

        """Return the input function to get the test data.
        Returns:
            (Input function, IteratorInitializerHook):
                - Function that returns (features, labels) when called.
                - Hook to initialise input iterator.
        """
        # iterator_initializer_hook = params.hooks['iterator_init_hook']
        iterator_initializer_hook = params.hooks['hook']

        def test_inputs():
            """Returns training set as Operations.
            Returns:
                (features, labels) Operations that iterate over the dataset
                on every evaluation
            """
            with tf.name_scope('test_data'):

                # Get data
                images, labels, _ = Experiment.get_dataset_provider(params).load_testing_data()

                # Define placeholders
                images_placeholder = tf.placeholder(images.dtype, images.shape)
                labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

                # Build dataset iterator
                dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))
                dataset = dataset.batch(params.batch_size)
                iterator = dataset.make_initializable_iterator()
                next_example, next_label = iterator.get_next()

                # Set runhook to initialize iterator
                iterator_initializer_hook.iterator_initializer_func = \
                    lambda sess: sess.run(iterator.initializer, feed_dict={images_placeholder: images,
                                                                           labels_placeholder: labels})
                return next_example, next_label

        # Return function and hook
        return test_inputs, iterator_initializer_hook

    # Define the training inputs
    @staticmethod
    def get_train_inputs_fn(params):

        """ Return the input function to get the training data.
        Returns:
            (Input function, IteratorInitializerHook):
                - Function that returns (features, labels) when called.
                - Hook to initialise input iterator.
        """
        # iterator_initializer_hook = params.hooks['iterator_init_hook']
        iterator_initializer_hook = params.hooks['hook']

        def train_inputs():

            """ Returns training set as Operations.
            Returns:
                (features, labels) Operations that iterate over the dataset
                on every evaluation
            """
            with tf.name_scope('training_data'):

                # Get data
                images, labels, _ = Experiment.get_dataset_provider(params).load_training_data()

                # Define placeholders
                images_placeholder = tf.placeholder(images.dtype, images.shape)
                labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

                # Build dataset iterator
                dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))
                dataset = dataset.repeat(None)  # Infinite iterations
                dataset = dataset.shuffle(buffer_size=Cifar10.NUM_TRAIN_IMAGES)
                dataset = dataset.batch(params.batch_size)
                iterator = dataset.make_initializable_iterator()
                next_example, next_label = iterator.get_next()

                # Set runhook to initialize iterator
                iterator_initializer_hook.iterator_initializer_func = \
                    lambda sess: sess.run(iterator.initializer, feed_dict={images_placeholder: images,
                                                                           labels_placeholder: labels})
                # Return batched (features, labels)
                return next_example, next_label

        # Return function and hook
        return train_inputs, iterator_initializer_hook
