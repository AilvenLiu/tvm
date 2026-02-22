# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""TVM Script APIs of TVM Python Package for TIRX builtin ops."""
from typing import TYPE_CHECKING, Any, Callable

from .ir_builder.tir import tirx as _ir_builder_tirx
from .ir_builder.tir.tirx import *  # pylint: disable=redefined-builtin,unused-wildcard-import,wildcard-import
from . import tir as _tir

if TYPE_CHECKING:
    # Statically expose all T-namespace attributes (cuda, ptx, nvshmem, nki,
    # meta_var, exec scope helpers, scalar ops, etc.) so that type checkers
    # and IDEs can resolve Tx.cuda.xxx, Tx.meta_var, etc.
    # At runtime these are copied dynamically via the globals() loop below.
    from .tir import *  # type: ignore[assignment]  # noqa: F811


def _is_buffer_or_region(x):
    from tvm.tir import Buffer, BufferRegion  # pylint: disable=import-outside-toplevel

    return isinstance(x, (Buffer, BufferRegion))


# Overload: one name, dispatch by argument types (expr vs buffer/region).
_cast_region = _ir_builder_tirx.cast
_cast_expr = _tir.cast


def cast(*args, **kwargs):
    if len(args) >= 2 and _is_buffer_or_region(args[0]) and _is_buffer_or_region(args[1]):
        return _cast_region(*args, **kwargs)
    return _cast_expr(*args, **kwargs)


_min_region = _ir_builder_tirx.min
_min_expr = _tir.min
_max_region = _ir_builder_tirx.max
_max_expr = _tir.max


def min(*args, **kwargs):  # pylint: disable=redefined-builtin
    if len(args) >= 2 and _is_buffer_or_region(args[0]) and _is_buffer_or_region(args[1]):
        return _min_region(*args, **kwargs)
    return _min_expr(*args, **kwargs)


def max(*args, **kwargs):  # pylint: disable=redefined-builtin
    if len(args) >= 2 and _is_buffer_or_region(args[0]) and _is_buffer_or_region(args[1]):
        return _max_region(*args, **kwargs)
    return _max_expr(*args, **kwargs)


_sqrt_region = _ir_builder_tirx.sqrt
_sqrt_expr = _tir.sqrt
_exp_region = _ir_builder_tirx.exp
_exp_expr = _tir.exp
_exp2_region = _ir_builder_tirx.exp2
_exp2_expr = _tir.exp2


def sqrt(*args, **kwargs):
    if len(args) >= 1 and _is_buffer_or_region(args[0]):
        return _sqrt_region(*args, **kwargs)
    return _sqrt_expr(*args, **kwargs)


def exp(*args, **kwargs):
    if len(args) >= 1 and _is_buffer_or_region(args[0]):
        return _exp_region(*args, **kwargs)
    return _exp_expr(*args, **kwargs)


def exp2(*args, **kwargs):
    if len(args) >= 1 and _is_buffer_or_region(args[0]):
        return _exp2_region(*args, **kwargs)
    return _exp2_expr(*args, **kwargs)


globals()["cast"] = cast
globals()["min"] = min
globals()["max"] = max
globals()["sqrt"] = sqrt
globals()["exp"] = exp
globals()["exp2"] = exp2

for _name in dir(_tir):
    if not _name.startswith("_") and _name not in globals():
        globals()[_name] = getattr(_tir, _name)


def __getattr__(name: str) -> Callable[..., Any]:
    if name.startswith("_"):
        raise AttributeError(f"module 'tvm.script.tirx' has no attribute {name!r}")

    from .ir_builder.tir.tirx import _to_region, f_insert
    from tvm.tir import Buffer
    from tvm.tirx.operator.op import GenericOp

    def _generic_op(*args, workspace=None, dispatch=None, **kwargs):
        workspace = workspace or {}
        config = kwargs or {}
        converted = [_to_region(a) if isinstance(a, Buffer) else a for a in args]
        return f_insert(
            GenericOp(
                *converted, op_name=name, workspace=workspace, config=config, dispatch=dispatch
            )
        )

    _generic_op.__name__ = name
    return _generic_op
