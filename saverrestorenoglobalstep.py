import os
import numpy as np
import tensorflow as tf
import six

from tensorpack.utils import logger
from tensorpack.tfutils.varmanip import (SessionUpdate, get_savename_from_varname,
                       is_training_name, get_checkpoint_path)
from tensorpack.tfutils.sessinit import (SessionInit, CheckpointReaderAdapter)


class SaverRestoreNoGlobalStep(SessionInit):
    """
    Restore a tensorflow checkpoint saved by :class:`tf.train.Saver` or :class:`ModelSaver`.
    """
    def __init__(self, model_path, prefix=None):
        """
        Args:
            model_path (str): a model name (model-xxxx) or a ``checkpoint`` file.
            prefix (str): during restore, add a ``prefix/`` for every variable in this checkpoint
        """
        model_path = get_checkpoint_path(model_path)
        self.path = model_path
        self.prefix = prefix


    def _setup_graph(self):
        dic = self._get_restore_dict()
        self.saver = tf.train.Saver(var_list=dic, name=str(id(dic)))

    def _run_init(self, sess):
        logger.info("Restoring checkpoint from {} ...".format(self.path))
        self.saver.restore(sess, self.path)

    @staticmethod
    def _read_checkpoint_vars(model_path):
        """ return a set of strings """
        reader = tf.train.NewCheckpointReader(model_path)
        reader = CheckpointReaderAdapter(reader)    # use an adapter to standardize the name
        ckpt_vars = reader.get_variable_to_shape_map().keys()
        return reader, set(ckpt_vars)

    def _match_vars(self, func):
        reader, chkpt_vars = SaverRestoreNoGlobalStep._read_checkpoint_vars(self.path)
        graph_vars = tf.global_variables()
        chkpt_vars_used = set()
        for v in graph_vars:
            name = get_savename_from_varname(v.name, varname_prefix=self.prefix)
            # skip global step
            if name == "global_step:0":
                print("skip restoring global step!")
                continue
            
            if reader.has_tensor(name):
                func(reader, name, v)
                chkpt_vars_used.add(name)
            else:
                vname = v.op.name
                if not is_training_name(vname):
                    logger.warn("Variable {} in the graph not found in checkpoint!".format(vname))
        if len(chkpt_vars_used) < len(chkpt_vars):
            unused = chkpt_vars - chkpt_vars_used
            for name in sorted(unused):
                if not is_training_name(name):
                    logger.warn("Variable {} in checkpoint not found in the graph!".format(name))

    def _get_restore_dict(self):
        var_dict = {}

        def f(reader, name, v):
            name = reader.get_real_name(name)
            assert name not in var_dict, "Restore conflict: {} and {}".format(v.name, var_dict[name].name)
            var_dict[name] = v
        self._match_vars(f)
        return var_dict
