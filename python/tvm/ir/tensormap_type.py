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
"""TensorMap type in the IR."""
from typing import List, Optional
from enum import IntEnum

import tvm
import tvm.ffi

from . import _ffi_api
from .type import Type


class TensorMapInterleaveKind(IntEnum):
    """Possible kinds of TensorMap interleaving."""

    kNone = 0
    k16B = 1
    k32B = 2


class TensorMapSwizzleKind(IntEnum):
    """Possible kinds of TensorMap swizzling."""

    kNone = 0
    k32B = 1
    k64B = 2
    k128B = 3
    k128B_BASE32B = 4


class TensorMapL2PromotionKind(IntEnum):
    """Possible kinds of TensorMap L2 promotion."""

    kNone = 0
    kL2_64B = 1
    kL2_128B = 2
    kL2_256B = 3


class TensorMapOOBFillKind(IntEnum):
    """Possible kinds of TensorMap out-of-bounds fill."""

    kNone = 0
    kNan = 1


@tvm.ffi.register_object("TensorMapType")
class TensorMapType(Type):
    """TensorMap type in the IR.

    Parameters
    ----------
    dtype : str
        The data type of the tensor.

    global_shape : List[int]
        The shape of the global tensor.

    global_strides : List[int]
        The strides of the global tensor.

    shared_shape : List[int]
        The shape of the shared memory tensor.

    shared_strides : List[int]
        The strides of the shared memory tensor.

    interleave : TensorMapInterleaveKind
        The interleaving kind.

    swizzle : TensorMapSwizzleKind
        The swizzling kind.

    l2_promotion : TensorMapL2PromotionKind
        The L2 promotion kind.

    oob_fill : TensorMapOOBFillKind
        The out-of-bounds fill kind.
    """

    def __init__(
        self,
        dtype: str,
        global_shape: List[int],
        global_strides: List[int],
        shared_shape: List[int],
        shared_strides: List[int],
        interleave: Optional[TensorMapInterleaveKind] = TensorMapInterleaveKind.kNone,
        swizzle: Optional[TensorMapSwizzleKind] = TensorMapSwizzleKind.kNone,
        l2_promotion: Optional[TensorMapL2PromotionKind] = TensorMapL2PromotionKind.kNone,
        oob_fill: Optional[TensorMapOOBFillKind] = TensorMapOOBFillKind.kNone,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.TensorMapType,
            dtype,
            global_shape,
            global_strides,
            shared_shape,
            shared_strides,
            interleave,
            swizzle,
            l2_promotion,
            oob_fill,
        )
