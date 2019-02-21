import tensorflow as tf

from nets.cifar10 import Cifar10

models = {
    'cifar-10': Cifar10
}


class Estimator:

    def __init__(self, run_config, params):
        self.run_config = run_config
        self.params = params

    def get_estimator(self):

        # You can change a subset of the run_config properties as
        # run_config = self.run_config.replace(save_checkpoints_steps=self.params.min_eval_frequency)

        # Create the Estimator
        return tf.estimator.Estimator(model_fn=self.get_model_provider(self.params.model).get_estimator_spec,
                                      config=self.run_config,
                                      params=self.params,
                                      model_dir=self.params.checkpoint_dir)

    def get_model_provider(self, model):
        if model not in models:
            raise ValueError('Name of model unknown %s' % model)

        return models[model](self.params)
