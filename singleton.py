#!/usr/bin/env python

import os
import sys
sys.path.append(os.getcwd())
import json
import time
import random

import numpy as np
import tensorflow as tf
import coref_model_gnc as cm
import util

if __name__ == "__main__":
  if len(sys.argv) > 1:
    name = sys.argv[1]
  else:
    name = os.environ["EXP"]
  config = util.get_config("experiments.conf")[name]
  report_frequency = config["report_frequency"]

  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))
  util.print_config(config)

  # if "GPU" in os.environ:
  #   util.set_gpus([0])
  # else:
  #   util.set_gpus([0])

  model = cm.CorefModel(config, 1)
  saver = tf.train.Saver()
  init_op = tf.global_variables_initializer()

  log_dir = config["log_dir"]
  writer = tf.summary.FileWriter(os.path.join(log_dir, "train"), flush_secs=20)

  # Create a "supervisor", which oversees the training process.
  sv = tf.train.Supervisor(logdir=log_dir,
                           init_op=init_op,
                           saver=saver,
                           global_step=model.global_step,
                           save_model_secs=120)

  # The supervisor takes care of session initialization, restoring from
  # a checkpoint, and closing when done or an error occurs.
  ii = 0
  with sv.managed_session() as session:
    model.start_enqueue_thread(session)
    # acc_mention_loss = 0.0
    acc_tagging_loss = 0.0
    initial_time = time.time()
    while not sv.should_stop():
      loss, tf_global_step, _, x1, x2, x3, x4, x5, x6, x8, x9 = session.run([model.loss,
                                              model.global_step,
                                              model.train_op,
                                              model.logits_shape,
                                              model.span_seq,
                                              model.p,
                                              model.fw_state,
                                              model.span_labels,
                                              model.tagging_loss,
                                              # model.mention_loss,
                                              model.antecedent_loss,
                                              model.antecedent_scores_shape])
      # acc_mention_loss += mention_loss
      acc_tagging_loss += loss
      '''
      ii += 1
      print '----------------------------'
      print x1
      print "number of entities:%d" % max(list(x2))
      print "tagging_loss:%f, mention_loss: NA, antecedent_loss:%f" % (x6, x8)
      print list(x2)
      print util.check_tags(x2)
      print list(x3)
      print x4
      print x9
      print '----------------------------'
      # print gs
      if ii == 2:
        print x + 1
      # '''

      # if tf_global_step % report_frequency == 0:
      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        # avg_mention_loss = acc_mention_loss / report_frequency
        avg_tagging_loss = acc_tagging_loss / report_frequency
        print "[{}] tagging_loss={:.2f} steps/s={:.2f}".format(tf_global_step,
                                                            avg_tagging_loss,
                                                            steps_per_second)
        # '''
        print '----------------------------'
        print x1
        print "number of entities:%d" % max(list(x2))
        print "tagging_loss:%f, mention_loss:NA, antecedent_loss:%f" % (x6, x8)
        print list(x2)
        print util.check_tags(x2)
        print list(x3)
        print x4
        print '----------------------------'
        # '''

        writer.add_summary(util.make_summary({"loss": avg_tagging_loss}), tf_global_step)
        # accumulated_loss = 0.0
        # acc_mention_loss = 0
        acc_tagging_loss = 0

  # Ask for all the services to stop.
  sv.stop()
