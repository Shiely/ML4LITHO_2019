import tensorflow as tf
from tensorboard.plugins.beholder import Beholder
from tensorboard.plugins.beholder import BeholderHook
from classes.SessionHook import SessionHook

import logging

def specs(train_input_fn, eval_input_fn, logdir='.', projectordir='.', max_train_steps=10000, eval_steps=1000,scopes=[],name='model'):
    logging.getLogger().setLevel(logging.INFO)  # to show info about training progress in the terminal
    beholder = Beholder(logdir)
    beholder_hook = BeholderHook(logdir)
    projector_hook = SessionHook(projectordir, scopes)
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps = max_train_steps, hooks=[beholder_hook])

    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=eval_steps,name= name + '-eval', throttle_secs=10, hooks=[projector_hook])
    return train_spec, eval_spec