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
# ruff: noqa: F401, RUF005

# pylint: disable=invalid-name,too-many-locals

"""Utility functions for Relax"""

import itertools
import string
from collections.abc import Callable
from typing import Any, Optional

import tvm_ffi

import tvm

from .. import tirx
from ..ir import Array, Attrs, Map, Type, VDevice
from ..te import Tensor as te_Tensor
from ..te import create_prim_func
from ..tirx import PrimExpr
from . import _ffi_api
from .expr import Expr, Function, PrimValue, ShapeExpr, StringImm, te_tensor
from .expr import Tuple as rx_Tuple
from .struct_info import PrimStructInfo, ShapeStructInfo, TensorStructInfo


def metadata_partitioner(rx_txt: str) -> list[str]:
    """Extract Relax program and metadata section.

    Parameters
    ----------
    rx_txt : str
        The input relax text.

    Returns
    -------
    output : List[str]
        The result list of partitioned text, the first element
        is the relax program, and the second is metadata section.
    """
    partitions = []
    left_curly = 0
    meta_start = 0
    meta_end = 0
    for i, char in enumerate(rx_txt):
        if i < 0:
            raise ValueError("The program is invalid.")
        if char == "{":
            if meta_start == 0:
                meta_start = i
            left_curly += 1
        elif char == "}":
            left_curly -= 1
            if left_curly == 0:
                meta_end = i + 1
                break

    if meta_end == 0:
        raise ValueError("The metadata section was not found.")
    metadata = rx_txt[meta_start:meta_end]
    rx_program = rx_txt[meta_end:-1]

    partitions.append(rx_program)
    partitions.append(metadata)

    return partitions


def convert_to_expr(value: Any) -> Expr:
    """Helper function to convert the input to Expr, which follows the rules:
    1. Return the input itself if it's already a `relax.Expr`;
    2. Return `relax.PrimValue` if the input is a `PrimExpr`;
    3. Return `relax.StringImm` if the input is `tvm.String` or `str`;
    4. Return `relax.Tuple` if the input is a tuple/list of `Expr`.

    Notes
    -----
    1. `tvm.tirx.StringImm` is not allowed because of ambiguity,
       which can be either `relax.StringImm` or `relax.PrimValue`.
    """
    if isinstance(value, int):
        return PrimValue(tirx.IntImm("int64", value))

    if isinstance(value, float):
        return PrimValue(tirx.FloatImm("float64", value))

    tvm_value = tvm_ffi.convert(value)
    # Case 1
    if isinstance(tvm_value, Expr):  # type: ignore
        return tvm_value
    # Note`` 1
    if isinstance(tvm_value, tirx.StringImm):
        raise TypeError(
            "Cannot convert `tirx.StringImm` to `relax.Expr` because of ambiguity,"
            "which can be either `relax.StringImm` or `relax.PrimValue` "
        )
    # Case 2
    if isinstance(tvm_value, PrimExpr):
        return PrimValue(value)
    # Case 3
    if isinstance(tvm_value, str):
        return StringImm(value)
    # Case 4
    if isinstance(value, tuple | list):
        # `convert_to_expr` ensures that all elements are `Expr` if no exception raises
        return rx_Tuple([convert_to_expr(v) for v in value])
    raise TypeError(f"Cannot convert {value} with type {type(value)} to `relax.Expr`")


def copy_with_new_vars(func: Function) -> Function:
    """Copy the given function. All variables that are bound inside the original function
    would be copied to satisfy the restriction in the well-formed check: Variables in
    Relax must be bound exactly once. This also ensures that both the function and its copy
    can be inserted into the same IRModule, and be asserted on the structural equality
    agaisnt IRModule created by TVMScript.

    Parameters
    ----------
    func : Function
        The relax function to copy.

    Returns
    -------
    ret : Function
        The copied function.
    """
    return _ffi_api.CopyWithNewVars(func)  # type: ignore


def gen_call_tir_inputs(
    func: Callable, *args: Any, **kwargs: Any
) -> tuple[tirx.PrimFunc, Expr, list[TensorStructInfo], ShapeExpr | None]:
    """Generate the inputs for call_tir according to the te function.
    This function converts arguments from relax expression to te tensor,
    The callback func should return a te tensor or a list of te tensors.

    Parameters
    ----------
    func : Callable
        A function that returns a te tensor or a list of te tensors.

    args : Any, optional
        arguments passed to the function.

    kwargs : Any, optional
        The keyword arguments passed to the function.
        Note that the keyword args 'primfunc_attrs' is reserved for passing func
        attributes to be added to the PrimFunc that gets created.

    Returns
    -------
    ret : Tuple[tirx.PrimFunc, Expr, List[TensorStructInfo], Optional[ShapeExpr]]
        ret contains the inputs for call_tir, including a tirx prim_func, args,
        out_sinfo, and tir_vars.
    """

    tir_var_map: dict[tirx.Var, tirx.PrimExpr] = {}

    call_tir_args = []
    create_primfunc_args = []
    # extra list of tirx expression arguments
    # that are not covered by Tensor
    extra_tir_args_list = []

    def _copy_undefined_var(expr: tirx.PrimExpr):
        def _visit_expr(e: tirx.PrimExpr):
            if isinstance(e, tirx.Var) and e not in tir_var_map:
                new_var = tirx.Var(e.name, e.dtype)
                tir_var_map[e] = new_var

        tirx.stmt_functor.post_order_visit(expr, _visit_expr)

    def _convert_te_arg(te_args: Any) -> Any:
        """Helper function used to convert Relax expressions to TE tensor.

        In the common case, the type of te_args is a Relax expression and is converted
        into a TE tensor.
        If te_args is a nested or recursive datatype (i.e list, dict, tvm.ir.Map, tvm.ir.Array),
        we recursive and convert any value of type Relax expression into a TE tensor.
        Common values of type int, float, and str are preserved.

        In dynamic shape cases, the passed in arguments may contain TIR variable.
        For example, the argument can be a Relax Var with TensorStructInfo, which
        has symbolic shape, or the argument can be a ShapeExpr with symbolic variables.
        To make the PrimFunc generated has independent variables with
        the caller Relax function, we will substitute the TIR variables in the input
        arguments with fresh ones, which is done by maintaining a TIR variable mapping.

        Parameters
        ----------
        te_args : Any
            Argument to convert to TE

        tir_var_map : Dict[tirx.Var, tirx.PrimExpr]
            The TIR variable mapping, which maps TIR variables on the Relax function
            side to the new set of variables used on the PrimFunc side.

        Returns
        -------
        ret : (Any, [tvm.te.Tensor])
            A tuple of the converted te_args, and a list of te tensors for each converted
            Relax expression
        """

        def _convert_te_arg_helper(arg):
            if isinstance(arg, Expr):  # type: ignore
                if isinstance(arg.struct_info, TensorStructInfo):
                    assert isinstance(arg.struct_info.shape, ShapeExpr), (
                        "emit_te now only supports Tensor that has ShapeExpr shape"
                    )
                    for shape_value in arg.struct_info.shape.values:
                        _copy_undefined_var(shape_value)

                    n_args = len(create_primfunc_args)
                    if isinstance(arg, tvm.relax.Var):
                        name = arg.name_hint
                    elif n_args < len(string.ascii_uppercase):
                        name = string.ascii_uppercase[n_args]
                    else:
                        name = f"tensor_input_{n_args}"

                    te_arg = te_tensor(arg, tir_var_map, name)

                    call_tir_args.append(arg)
                    create_primfunc_args.append(te_arg)

                    return te_arg

                if isinstance(arg.struct_info, ShapeStructInfo):
                    assert isinstance(arg, ShapeExpr), (
                        "For Expr having ShapeStructInfo, emit_te now only supports ShapeExpr"
                    )
                    return [_convert_te_arg_helper(val) for val in arg.values]

                if isinstance(arg.struct_info, PrimStructInfo):
                    if arg.struct_info.value is None:
                        n_args = len(create_primfunc_args)
                        if isinstance(arg, tvm.relax.Var):
                            name = arg.name_hint
                        elif n_args < len(string.ascii_lowercase):
                            name = string.ascii_lowercase[n_args]
                        else:
                            name = f"scalar_input_{n_args}"

                        tir_param = tirx.Var(name, arg.struct_info.dtype)

                        call_tir_args.append(arg)
                        create_primfunc_args.append(tir_param)

                        return tir_param
                    else:
                        return _convert_te_arg_helper(arg.struct_info.value)

            elif isinstance(arg, list | Array):
                return [_convert_te_arg_helper(x) for x in arg]
            elif isinstance(arg, tuple):
                return tuple(_convert_te_arg_helper(x) for x in arg)
            elif isinstance(arg, dict | Map):
                for key in arg:
                    assert isinstance(key, str), (
                        "emit_te only supports dict with string as the key currently"
                    )
                return {k: _convert_te_arg_helper(arg[k]) for k in arg}
            elif isinstance(arg, tirx.PrimExpr):
                _copy_undefined_var(arg)
                new_arg = tirx.stmt_functor.substitute(arg, tir_var_map)
                extra_tir_args_list.append(new_arg)
                return new_arg
            elif isinstance(arg, int | float | str | Type | Attrs) or arg is None:
                return arg
            raise TypeError(f"not supported type in emit_te: {type(arg)}")

        new_arg = _convert_te_arg_helper(te_args)
        return new_arg

    def _get_unbound_tir_vars(
        args: list[te_Tensor], extra_tir_args: list[PrimExpr]
    ) -> list[tirx.Var]:
        """get unbound TIR vars (i.e TIR vars used in the shape but is not
        itself a dimension of a shape)"""

        bound_vars = set()
        used_vars = set()

        def _populate_bound_vars(expr):
            if isinstance(expr, te_Tensor):
                for dim in expr.shape:
                    _populate_bound_vars(dim)
            elif isinstance(expr, tirx.Var):
                bound_vars.add(expr)

        def _populate_used_vars(expr):
            if isinstance(expr, te_Tensor):
                for dim in expr.shape:
                    _populate_used_vars(dim)
            elif isinstance(expr, tirx.PrimExpr):
                used_vars.update(tirx.analysis.undefined_vars(expr))

        for arg in itertools.chain(args, extra_tir_args):
            _populate_used_vars(arg)

        for arg in args:
            _populate_bound_vars(arg)

        diff = used_vars - bound_vars
        return list(diff)

    def _get_vdevice(arg: Any) -> VDevice | None:
        """get the virtual device from arguments."""
        vdevice = None
        if isinstance(arg, Expr):  # type: ignore
            if isinstance(arg.struct_info, TensorStructInfo):
                vdevice = arg.struct_info.vdevice
        elif isinstance(arg, list | Array | tuple):
            for x in arg:
                vdevice = _get_vdevice(x)
                if vdevice is not None:
                    return vdevice
        elif isinstance(arg, dict | Map):
            for k in arg:
                vdevice = _get_vdevice(arg[k])
                if vdevice is not None:
                    return vdevice
        return vdevice

    def _shape_with_old_tir_var(
        shape_values: list[tirx.PrimExpr], tir_var_inverse_map: dict[tirx.Var, tirx.PrimExpr]
    ):
        return ShapeExpr(
            [tirx.stmt_functor.substitute(value, tir_var_inverse_map) for value in shape_values]
        )

    primfunc_attrs = kwargs.pop("primfunc_attrs", None)
    custom_out_sinfo = kwargs.pop("sinfo_args", [])

    te_args = _convert_te_arg(args)
    te_kwargs = _convert_te_arg(kwargs)

    te_out = func(*te_args, **te_kwargs)
    assert isinstance(te_out, te_Tensor) or (
        isinstance(te_out, tuple | list | Array) and all(isinstance(t, te_Tensor) for t in te_out)
    ), "only support te.tensor or tuple/list/Array of te.tensor as function output"

    outs = [te_out] if isinstance(te_out, te_Tensor) else list(te_out)
    unbound_tir_vars = _get_unbound_tir_vars([*create_primfunc_args, *outs], extra_tir_args_list)

    inputs = [*create_primfunc_args] + outs + unbound_tir_vars
    tir_func = create_prim_func(inputs, "int64")

    if primfunc_attrs:
        tir_func = tir_func.with_attrs(primfunc_attrs)

    tir_func = tir_func.without_attr("global_symbol")

    # Invert the TIR variable mapping, to convert the output shape back
    # with old set of variables.
    tir_var_inverse_map = {v: k for k, v in tir_var_map.items()}

    if len(custom_out_sinfo) == 1:
        output_sinfo = custom_out_sinfo[0]
    else:
        output_sinfo = [
            TensorStructInfo(
                _shape_with_old_tir_var(out.shape, tir_var_inverse_map),
                out.dtype,
                _get_vdevice(args),
            )
            for out in outs
        ]

    tir_vars = None
    if len(unbound_tir_vars) > 0:
        tir_vars = _shape_with_old_tir_var(unbound_tir_vars, tir_var_inverse_map)

    return (tir_func, call_tir_args, output_sinfo, tir_vars)

from typing import Tuple, Union
import inspect
from tvm.script import tir as T
from tvm.tir.function import PrimFunc
ImmeValueType = Union[int, float]
PrimExprLike = Union[ImmeValueType, PrimExpr]

class Dependency:
    def __init__(
        self,
        event: Expr,
        dep: Callable,
        num: Union[Callable, int],
        extra_args: List[Union[Expr, PrimExprLike]] = [],
        tile_idx_dtype: str = "int32",
    ):
        """
        Parameters
        ----------
        event : Expr
            The variable of event tensor.
        dep : Callable
            The dependency function. 
            It takes the (tile index/event coordinate), the index of the (event coordinate/tile index) to be calculated, 
            and extra arguments, and returns the indices of the depended (event coordinate/tile index).
        num : Union[Callable, int]
            The number of depended (event coordinates/tile indices).
            This can be a fixed integer or a Callable that dynamically computes this number.
        extra_args : List[Union[Expr, PrimExprLike]], optional
            A list of extra arguments passed to the dependency function.
            Defaults to an empty list.
        tile_idx_dtype : str, optional
            The data type of the tile index. Defaults to "int32".

        Notes
        -----
        The order of parameters for both callable functions is critical and strictly defined.
        1. The dependency function (`dep`) must accept parameters in the following order:
           The (tile index/event coordinate),
           The index of the (event coordinate/tile index) to be calculated (ranging from 0 to num - 1),
           The extra arguments in the same order as provided in `extra_args`.
        2. The function to calculate the number of coordinates (if 'num' is Callable) must accept parameters in the following order:
           The (tile index/event coordinate),
           The extra arguments in the same order as provided in `extra_args`.
        3. The event coordinate in defined as (rank, *evt_tensor_indices). 'rank=-1' represents the local rank.
           
        Examples
        --------
        Task (i, j, k) depends on events E at coordinates (rank=-1, X[i], j, 0), (rank=-1, X[i], j, 1), ..., (rank=-1, X[i], j, Y[k]),
        where X, Y are the tensors.
        Then the Dependency object can be created as follows:
        >>> dependency = Dependency(
        ...     event=E,
        ...     dep=lambda i, j, k, idx, X, Y: (-1, X[i], j, idx),
        ...     num=lambda i, j, k, X, Y: Y[k] + 1,
        ...     extra_args=[X, Y],
        ... )
        Then the dependency can be handled by:
        >>> handled_info = dependency.handle_dep(input_dim=3, output_dim=4)
        """
        self.event = event
        if not isinstance(event, Expr) or isinstance(event, List) or isinstance(event, Tuple) or isinstance(event, tvm.relax.Tuple):
            raise ValueError("One Dependency obj should handle a single event.")
        self.dep = dep
        if isinstance(num, int):
            self.num = lambda *args: num
            self.pass_num_signature_check = True
        else:
            self.num = num
            self.pass_num_signature_check = False

        self.extra_args = extra_args
        for arg in self.extra_args:
            if not isinstance(arg, (Expr, PrimExprLike)):
                raise ValueError("extra_args should be Expr or PrimExpr/int/float")
        self.tile_idx_dtype = tile_idx_dtype
        

    def handle_dep(self, input_dim, output_dim) -> Tuple[Expr, List[Expr], List[PrimExpr], PrimFunc, PrimFunc, List[Optional[ImmeValueType]], List[Optional[ImmeValueType]]]:
        # check the signature of dep and num
        if len(inspect.signature(self.dep).parameters) != input_dim + 1 + len(self.extra_args):
             raise ValueError(f"dep requires {input_dim + 1 + len(self.extra_args)} parameters")
        if not self.pass_num_signature_check and len(inspect.signature(self.num).parameters) != input_dim + len(self.extra_args):
             raise ValueError(f"num requires {input_dim + len(self.extra_args)} parameters")

        # create dep and num param
        dep_input = [T.var(dtype=self.tile_idx_dtype) for _ in range(input_dim + 1)]
        for arg in self.extra_args:
            if isinstance(arg, Expr):
                dep_input.append(T.Buffer(shape=[v for v in arg.struct_info.shape], dtype=arg.struct_info.dtype, scope="global"))
            elif isinstance(arg, PrimExpr):
                dep_input.append(T.var(dtype=arg.dtype))
            else: # int or float
                dep_input.append(arg)
        dep_output = [T.Buffer(shape=[1], dtype=self.tile_idx_dtype, scope="local") for _ in range(output_dim)]
        
        num_input = [T.var(dtype=self.tile_idx_dtype) for _ in range(input_dim)]
        for arg in self.extra_args:
            if isinstance(arg, Expr):
                num_input.append(T.Buffer(shape=[v for v in arg.struct_info.shape], dtype=arg.struct_info.dtype, scope="global"))
            elif isinstance(arg, PrimExpr):
                num_input.append(T.var(dtype=arg.dtype))
            else: # int or float
                num_input.append(arg)
        num_output = [T.Buffer(shape=[1], dtype=self.tile_idx_dtype, scope="local")]

        # pack dep and num func to PrimFunc
        def _convert_arg(para: Union[tir.Var, tir.Buffer]) -> T.Var:
            if isinstance(para, tir.Var) or isinstance(para, tir.Buffer):
                return T.arg("var", para)
            else:
                return para

        def _unpack_and_assign(value_list: List[T.Var], buf_list: List[T.Buffer], length):
            if length == 1:
                T.buffer_store(buf_list[0], value_list, [0])
            else:
                for i in range(length):
                    T.buffer_store(buf_list[i], value_list[i], [0])

        @T.prim_func(check_well_formed=False)
        def dep():
            in_args = T.meta_var([_convert_arg(spec) for spec in dep_input])
            out_arg = T.meta_var([_convert_arg(spec) for spec in dep_output])
            with T.cta():
                _unpack_and_assign(self.dep(*in_args), out_arg, output_dim)
        
        @T.prim_func(check_well_formed=False)
        def num():
            in_args = T.meta_var([_convert_arg(spec) for spec in num_input])
            out_arg = T.meta_var([_convert_arg(spec) for spec in num_output])
            with T.cta():
                _unpack_and_assign(self.num(*in_args), out_arg, 1)

        return self.event, [var for var in self.extra_args if isinstance(var, (Expr, PrimExpr))], dep, num