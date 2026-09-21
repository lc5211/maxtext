# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit logical-axis to mesh-axis sharding rules for Cosmos 3."""

from __future__ import annotations

import abc
from collections.abc import Callable
import enum
import itertools
from typing import Any, TypeAlias

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


class TensorType(enum.Enum):
  """Semantic role of a tensor for sharding rule lookup."""

  WEIGHT = "weight"
  ACTIVATION = "activation"
  INPUT = "input"
  OUTPUT = "output"


MeshAxis: TypeAlias = str
MeshAxisMapping: TypeAlias = MeshAxis | tuple[MeshAxis, ...] | None
LogicalAxis: TypeAlias = str | None


def _flatten_mesh_axes(mesh_axes: MeshAxisMapping) -> tuple[MeshAxis, ...]:
  """Flattens nested mesh axis tuples into a flat tuple of axis names."""
  if mesh_axes is None:
    return ()
  if isinstance(mesh_axes, tuple):
    return tuple(itertools.chain.from_iterable(_flatten_mesh_axes(axis) for axis in mesh_axes))
  return (mesh_axes,)


class Sharding(abc.ABC):
  """Maps semantic logical tensor axes to explicit mesh PartitionSpecs."""

  @abc.abstractmethod
  def map_axis(
      self,
      axis: str,
      tensor_name: str,
      tensor_type: TensorType,
  ) -> MeshAxisMapping:
    """Maps one logical axis to physical mesh axis name(s), or None."""

  def spec(
      self,
      tensor_name: str,
      axes: tuple[LogicalAxis, ...],
      tensor_type: TensorType = TensorType.WEIGHT,
  ) -> P:
    """Builds a PartitionSpec for the given logical axes."""
    return P(*(None if axis is None else self.map_axis(axis, tensor_name, tensor_type) for axis in axes))

  def __call__(
      self,
      tensor_name: str,
      axes: tuple[LogicalAxis, ...],
      tensor_type: TensorType = TensorType.WEIGHT,
  ) -> P:
    return self.spec(tensor_name, axes, tensor_type)

  def weight_spec(self, tensor_name: str, axes: tuple[LogicalAxis, ...]) -> P:
    """Builds a weight PartitionSpec."""
    return self.spec(tensor_name, axes, TensorType.WEIGHT)

  def activation_spec_for(self, tensor_name: str, axes: tuple[LogicalAxis, ...]) -> P:
    """Builds an activation PartitionSpec."""
    return self.spec(tensor_name, axes, TensorType.ACTIVATION)

  def init_weight_spec(
      self,
      tensor_name: str,
      in_axes: tuple[LogicalAxis, ...],
      out_axes: tuple[LogicalAxis, ...],
  ) -> P:
    """Builds an initializer-time 2D PartitionSpec for flattened linear weights."""

    def collapse(axes: tuple[LogicalAxis, ...]) -> MeshAxisMapping:
      mapped = [self.map_axis(axis, tensor_name, TensorType.WEIGHT) for axis in axes if axis is not None]
      mapped = [axis for axis in mapped if axis is not None]
      if not mapped:
        return None
      flattened = tuple(itertools.chain.from_iterable(_flatten_mesh_axes(axis) for axis in mapped))
      if len(flattened) == 1:
        return flattened[0]
      return flattened

    return P(collapse(in_axes), collapse(out_axes))


class Cosmos3Sharding(Sharding):
  """Logical-to-mesh sharding rules for Cosmos 3 dual-pathway Mixture-of-Transformers."""

  def map_axis(
      self,
      axis: str,
      tensor_name: str,
      tensor_type: TensorType,
  ) -> MeshAxisMapping:
    del tensor_name
    match axis, tensor_type:
      case "batch", _:
        return ("dp", "fsdp")
      case "embed", TensorType.WEIGHT:
        return ("fsdp", "fsdp_t")
      case "embed", _:
        return None
      case "norm", _:
        return "fsdp_t"
      case ("heads" | "q_heads" | "kv_heads" | "qkv_heads"), _:
        return ("tp", "tp_t")
      case ("mlp" | "expert_mlp"), _:
        return ("tp", "tp_t", "tp_s")
      case ("head_dim" | "kv" | "length" | "kv_length"), _:
        return None
      case "norm_length", _:
        return ("sp", "cp")
      case "vocab", _:
        return ("tp", "tp_s")
      case _, _:
        raise ValueError(f"Unexpected logical axis name for Cosmos3Sharding: {axis=} {tensor_type=}")


def _has_active_mesh() -> bool:
  """Returns True when a non-empty JAX concrete or abstract mesh is active."""
  mesh = jax.sharding.get_abstract_mesh()
  return mesh is not None and not getattr(mesh, "empty", not bool(mesh.shape))


def sharded_init(init_fn: Callable[..., jax.Array], sharding_spec: P) -> Callable[..., jax.Array]:
  """Wraps a weight initializer to place the initialized array on ``sharding_spec``."""

  def init(*args: Any, **kwargs: Any) -> jax.Array:
    x = init_fn(*args, **kwargs)
    if not _has_active_mesh():
      return x
    return jax.device_put(x, sharding_spec)

  return init


def sharded_constant_init(value: float | int, sharding_spec: P) -> Callable[..., jax.Array]:
  """Creates a constant initializer placed on ``sharding_spec``."""

  def init(_key: jax.Array, shape: tuple[int, ...], dtype: Any = jnp.float32) -> jax.Array:
    if value == 0:
      x = jnp.zeros(shape, dtype=dtype)
    elif value == 1:
      x = jnp.ones(shape, dtype=dtype)
    else:
      x = jnp.full(shape, value, dtype=dtype)
    if not _has_active_mesh():
      return x
    return jax.device_put(x, sharding_spec)

  return init


class Cosmos3ShardingHook:
  """Callable sharding hook for Cosmos 3 modules."""

  def __init__(self, sharding_rules: Sharding | None = None):
    self.sharding = sharding_rules or Cosmos3Sharding()

  def get_spec(self, name: str) -> NamedSharding | None:
    """Returns a NamedSharding for the given activation tag if an abstract mesh is active."""
    mesh = jax.sharding.get_abstract_mesh()
    if mesh is None or not mesh.shape:
      return None
    s = self.sharding
    match name:
      case "query" | "gen_query" | "attn_out" | "gen_attn_out":
        return NamedSharding(mesh, s.activation_spec_for(name, ("length", "heads", "head_dim")))
      case "key" | "value" | "gen_key" | "gen_value":
        return NamedSharding(mesh, s.activation_spec_for(name, ("kv_length", "kv_heads", "head_dim")))
      case "mlpwi" | "gen_mlpwi":
        return NamedSharding(mesh, s.activation_spec_for(name, ("length", "mlp")))
      case "post_attn" | "gen_post_attn" | "post_mlp" | "gen_post_mlp" | "attn_input" | "mlp_input":
        return NamedSharding(mesh, s.activation_spec_for(name, ("norm_length", "embed")))
      case _:
        return None

  def __call__(self, tensor_or_init: Any, name: str) -> Any:
    """Annotates an initializer or activation tensor with its logical sharding spec."""
    s = self.sharding
    match name:
      case "q_proj_kernel_init" | "q_proj_gen_kernel_init":
        return sharded_init(tensor_or_init, s.init_weight_spec("q_proj", ("embed",), ("heads", "head_dim")))
      case "k_proj_kernel_init" | "k_proj_gen_kernel_init":
        return sharded_init(tensor_or_init, s.init_weight_spec("k_proj", ("embed",), ("kv_heads", "head_dim")))
      case "v_proj_kernel_init" | "v_proj_gen_kernel_init":
        return sharded_init(tensor_or_init, s.init_weight_spec("v_proj", ("embed",), ("kv_heads", "head_dim")))
      case "o_proj_kernel_init" | "o_proj_gen_kernel_init":
        return sharded_init(tensor_or_init, s.init_weight_spec("o_proj", ("heads", "head_dim"), ("embed",)))
      case "gate_proj_kernel_init" | "up_proj_kernel_init":
        return sharded_init(tensor_or_init, s.init_weight_spec("up_proj", ("embed",), ("mlp",)))
      case "down_proj_kernel_init":
        return sharded_init(tensor_or_init, s.init_weight_spec("down_proj", ("mlp",), ("embed",)))
      case "qk_norm_scale_init":
        return sharded_constant_init(tensor_or_init, s.weight_spec("qk_norm", ("head_dim",)))
      case "norm_scale_init":
        return sharded_constant_init(tensor_or_init, s.weight_spec("norm", ("norm",)))
      case _:
        spec = self.get_spec(name)
        if spec is not None and isinstance(tensor_or_init, jax.Array):
          return jax.lax.with_sharding_constraint(tensor_or_init, spec)
        return tensor_or_init


def shard_cosmos3_parameters(model: nnx.Module, sharding_rules: Sharding | None = None) -> None:
  """Shards all ``nnx.Param`` leaves in a Cosmos 3 module using ``sharding_rules``."""
  rules = sharding_rules or Cosmos3Sharding()
  _, state = nnx.split(model, nnx.Param)
  flat = state.flat_state()
  sharded_flat = {}
  for path, param in zip(flat.paths, flat.leaves):
    axes = getattr(param, "kernel_axes", None) or getattr(param, "out_sharding", None)
    if isinstance(axes, tuple) and axes:
      if len(axes) == 2 and param.value.ndim == 2:
        spec = rules.init_weight_spec("/".join(str(p) for p in path), (axes[0],), (axes[1],))
      else:
        spec = rules.weight_spec("/".join(str(p) for p in path), axes)
      sharded_flat[path] = nnx.Param(jax.device_put(param.value, spec), kernel_axes=axes, sharding=axes)
    else:
      sharded_flat[path] = param
  nnx.update(model, nnx.State.from_flat_path(sharded_flat))
