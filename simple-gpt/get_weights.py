import tensorflow as tf
import json
import numpy as np

def get_gpt2_weights(dir):
    tf_checkpoint_path = tf.train.latest_checkpoint(dir)
    settings = json.load(open(dir + "hparams.json"))
    params = load_gpt2_params_from_tf_checkpoint(tf_checkpoint_path, settings)

    return settings, params

def load_gpt2_params_from_tf_checkpoint(checkpoint_path, settings):
    params = {"blocks": [{} for _ in range(settings['n_layer'])]}

    for name, _ in tf.train.list_variables(checkpoint_path):
        variable_array = np.squeeze(tf.train.load_variable(checkpoint_path, name))

        variable_name_parts = name.split("/")[1:]

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params