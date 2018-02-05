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

  model = cm.CorefModel(config)
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
  with sv.managed_session() as session:
    model.start_enqueue_thread(session)
    acc_mention_loss = 0.0
    acc_tagging_loss = 0.0
    initial_time = time.time()
    while not sv.should_stop():
      mention_loss, tagging_loss, tf_global_step, _, nm, ngm, nw, tag_outputs, tag_seq, cluster_ids, gs = session.run([model.mention_loss,
                                                            model.tagging_loss,
                                                            model.global_step,
                                                            model.train_op,
                                                            model.num_mention,
                                                            model.num_gold_mention,
                                                            model.num_words,
                                                            model.tag_outputs,
                                                            model.tag_seq,
                                                            model.cluster_ids,
                                                            model.gold_starts])
      acc_mention_loss += mention_loss
      acc_tagging_loss += tagging_loss
      # print cluster_ids
      # print gs
      # print x + 1

      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        avg_mention_loss = acc_mention_loss / report_frequency
        avg_tagging_loss = acc_tagging_loss / report_frequency
        print "[{}] mention_loss={:.2f}, tagging_loss={:.2f} steps/s={:.2f}, nm={}, ngm={}, nw={}".format(tf_global_step,
                                                            avg_mention_loss,
                                                            avg_tagging_loss,
                                                            steps_per_second,
                                                            nm,
                                                            ngm,
                                                            nw)
        print tag_outputs
        print tag_seq
        writer.add_summary(util.make_summary({"mention_loss": avg_mention_loss, "tagging_loss": avg_tagging_loss}), tf_global_step)
        # accumulated_loss = 0.0
        acc_mention_loss = 0
        acc_tagging_loss = 0

  # Ask for all the services to stop.
  sv.stop()
