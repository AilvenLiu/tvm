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
# pylint: disable=unused-import
"""CUDA HW ops package.

This package contains the codegen functions for CUDA hardware operations.
The ops are split into logical groups:

- header.py: CUDA header generator and tags
- registry.py: Codegen registry functions
- types.py: PTX data types
- mma.py: PTX MMA operations (Volta/Turing/Ampere)
- wgmma.py: PTX WGMMA operations (Hopper)
- tcgen05.py: PTX tcgen05 operations (Blackwell)
- barrier.py: PTX barrier and mbarrier operations
- cp_async.py: PTX cp.async and TMA operations
- misc_ptx.py: PTX miscellaneous operations
- timer.py: CUDA timer operations
- cuda_sync.py: CUDA C++ synchronization and misc operations
- nvshmem.py: NVSHMEM operations
"""

# Import header generator and tags (registers tir.device_op_codegen.cuda.header_generator)
# Import op modules to register their codegen functions
from . import barrier, cp_async, cuda_sync, math, misc_ptx, mma, nvshmem, tcgen05, timer, wgmma
from .header import TAGS, header_generator

# Import registry (registers tir.device_op_codegen.cuda.get_codegen)
from .registry import CODEGEN_REGISTRY, get_codegen, register_codegen

# Import types
from .types import PTXDataType

# Re-export commonly used items
__all__ = [
    "CODEGEN_REGISTRY",
    "TAGS",
    "PTXDataType",
    "get_codegen",
    "header_generator",
    "register_codegen",
]
