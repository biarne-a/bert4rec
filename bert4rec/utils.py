import inspect

import tensorflow as tf


def clone_initializer(initializer):
  # Keras initializer is going to be stateless, which mean reusing the same
  # initializer will produce same init value when the shapes are the same.
  if isinstance(initializer, tf.keras.initializers.Initializer):
    return initializer.__class__.from_config(initializer.get_config())
  # When the input is string/dict or other serialized configs, caller will
  # create a new keras Initializer instance based on that, and we don't need to
  # do anything
  return initializer


def serialize_initializer(initializer, use_legacy_format=False):
  if (
      "use_legacy_format"
      in inspect.getfullargspec(tf.keras.initializers.serialize).args
  ):
    return tf.keras.initializers.serialize(
        initializer, use_legacy_format=use_legacy_format
    )
  else:
    return tf.keras.initializers.serialize(initializer)


def serialize_regularizer(regularizer, use_legacy_format=False):
  if (
      "use_legacy_format"
      in inspect.getfullargspec(tf.keras.regularizers.serialize).args
  ):
    return tf.keras.regularizers.serialize(
        regularizer, use_legacy_format=use_legacy_format
    )
  else:
    return tf.keras.regularizers.serialize(regularizer)


def serialize_constraint(constraint, use_legacy_format=False):
  if (
      "use_legacy_format"
      in inspect.getfullargspec(tf.keras.constraints.serialize).args
  ):
    return tf.keras.constraints.serialize(
        constraint, use_legacy_format=use_legacy_format
    )
  else:
    return tf.keras.constraints.serialize(constraint)
