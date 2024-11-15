# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generates a toy JAX2TF saved model for testing."""

from collections.abc import Sequence

from absl import app
from absl import flags
from jax.experimental import jax2tf
import jax.numpy as jnp

from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.saved_model import save


_SAVED_MODEL_PATH = flags.DEFINE_string(
    'saved_model_path', '', 'Path to save the model to.'
)


def generate(output_dir: str) -> None:
  """Generates a SavedModel for a*x + b using JAX2TF (saves the CPU model only)."""

  def linear_combination(a, x, b):
    return jnp.dot(a, x) + b, 1

  @polymorphic_function.function(
      input_signature=[
          tensor_spec.TensorSpec([], dtypes.float32, name='a'),
          tensor_spec.TensorSpec([], dtypes.float32, name='x'),
          tensor_spec.TensorSpec([], dtypes.float32, name='b'),
      ],
      jit_compile=True,
      autograph=False,
  )
  def tf_linear_combination(a, x, b=constant_op.constant(0.0)):
    f = jax2tf.convert(
        linear_combination,
        native_serialization_platforms=['tpu'],
    )
    return f(a, x, b)

  def save_cpu_model(output_dir):
    model = module.Module()
    model.linear_combination = tf_linear_combination
    save.save(
        model,
        output_dir,
        signatures={
            'linear_combination': model.linear_combination,
        },
    )

  save_cpu_model(output_dir)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  generate(_SAVED_MODEL_PATH.value)


if __name__ == '__main__':
  app.run(main)
