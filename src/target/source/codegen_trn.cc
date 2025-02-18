/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file codegen_trn.cc
 */
#include <tvm/tir/transform.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "../../runtime/thread_storage_scope.h"
#include "../build_common.h"
#include "codegen_trn.h"

namespace tvm {
namespace codegen {

void CodeGenTrainium::InitFuncState(const PrimFunc& f) {
  CodeGenC::InitFuncState(f);
}

CodeGenTrainium::CodeGenTrainium(Target target) : target_(target) {
  decl_stream << "import neuronxcc.nki.language as nl\n";
  decl_stream << "from neuronxcc.nki import baremetal, benchmark, simulate_kernel, trace\n";
  decl_stream << "import numpy as np\n";
  decl_stream << "import neuronxcc.nki.isa as nisa\n";
  decl_stream << "import math\n";
  decl_stream << "import neuronxcc.nki as nki\n";
  decl_stream << "import neuronxcc.nki.typing as nt\n";
  decl_stream << "@nki.compiler.enable_stack_allocator\n";
  decl_stream << "@nki.compiler.skip_middle_end_transformations\n";
  decl_stream << "@baremetal(experimental_flags='enable-mutable-parameter')\n";
  opcode_map_ = {{"sqrt", "nki.language.sqrt"},
                 {"add", "nki.language.add"},
                 {"sub", "nki.language.subtract"},
                 {"mul", "nki.language.multiply"}};
}

void CodeGenTrainium::AddFunction(const GlobalVar& gvar, const PrimFunc& func) {
  // NOTE: There is no inter-function calls among Trainium kernels.
  // For now we keep the Trainium codegen without inter-function call
  // process.
  // We can switch to follow the flow with inter-function call process
  // after the Trainium function declaration is properly printed.
  // In Trainium, for PrimFuncs with signature
  //    def func(A: Buffer, B: Buffer, x: int, y: float) -> None
  // where there are trailing pod parameters, the codegen emits a struct
  //    struct func_params{ x: int; y: float; }
  // for the function. In the flow of inter-function call process,
  // the struct will be emitted for every time a function is declared.
  // So consequently there are duplicate appearances of a same struct,
  // which makes the Trainium compiler unable to recognize.

  // clear previous generated state.
  this->InitFuncState(func);
  // skip the first underscore, so SSA variable starts from _1
  name_supply_->FreshName("v_");

  // add to alloc buffer type.
  auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";

  // Function header.
  this->stream << "def " << static_cast<std::string>(global_symbol.value()) << "(";

  // Buffer arguments
  auto num_inputs = func->GetAttr<Integer>(tvm::attr::kNumInputs);
  ICHECK(num_inputs.defined());
  std::vector<std::string> output_vids;
  size_t num_buffer = 0;
  for (size_t i = 0; i < func->params.size(); ++i, ++num_buffer) {
    Var v = func->params[i];
    if (!v.dtype().is_handle()) {
      LOG(FATAL) << "Trainium codegen currently only support buffer arguments";
    };
    std::string vid = AllocVarID(v.get());
    if(i >= static_cast<size_t>(num_inputs.value()->value)){
      this->stream << vid << ": nt.mutable_tensor, ";
      output_vids.push_back(vid);
    } else {
      this->stream << vid << ", ";
    }
  }

  // the function scope.
  stream << "):\n";
  int func_scope = this->BeginScope();
  this->PrintStmt(func->body);
  this->PrintIndent();
  stream << "return ";
  for(size_t i = 0; i < output_vids.size(); i++){
    if(i != 0){
      stream << ", ";
    }
    stream << output_vids[i];
  }
  this->EndScope(func_scope);
}

void CodeGenTrainium::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  ICHECK(lanes == 1) << "Trainium codegen does not support vector types";
  ICHECK(!t.is_handle()) << "Trainium codegen does not support handle type";
  ICHECK(!t.is_void()) << "Trainium codegen does not support void type";
  if (t == DataType::Bool()) {
    os << "np.bool";
    return;
  }
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "np.float16";
        break;
      case 32:
        os << "np.float32";
        break;
      default:
        LOG(FATAL)<<"Trainium codegen does not support float type with bits " << t.bits();
        break;
    }
    return;
  }
  if (t.is_uint() || t.is_int()) {
    if(t.bits() == 1){
      os << "np.bool";
      return;
    }
    os << "np.";
    if (t.is_uint()) {
      os << 'u';
    }
    switch (t.bits()) {
      case 8:
        os << "int8";
        break;
      case 16:
        os << "int16";
        break;
      case 32:
        os << "int32";
        break;
      case 64:
        os << "int64";
        break;
      default:
        LOG(FATAL)<< "Trainium codegen does not support int type with bits " << t.bits();
        break;
    }
    return;
  } 
  if (t.is_bfloat16()) {
    os << "nl.bfloat16";
    return;
  }
  LOG(FATAL) << "Cannot convert type " << t << " to Trainium type";
}

std::string CodeGenTrainium::GetStorageScopeStr(const std::string& scope) {  // NOLINT(*)
  if (scope == "global") {
    return "nl.hbm";
  } else if (scope == "trn.sbuf") {
    return "nl.sbuf";
  } else if (scope == "trn.psum") {
    return "nl.psum";
  } else {
    LOG(FATAL) << "Unknown storage scope `" << scope << "`";
    return "";
  }
}

void CodeGenTrainium::VisitStmt_(const AllocateNode* op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  this->PrintIndent();
  auto scope = GetPtrStorageScope(op->buffer_var);
  std::ostringstream dtype_os;
  PrintType(op->dtype, dtype_os);
  std::string dtype_str = dtype_os.str();
  if (scope == "trn.psum") {
    stream << vid << " = nl.ndarray(shape=[";
    ICHECK(op->extents.size() == 3);
    stream << PrintExpr(op->extents[0]) << ", nl.par_dim(" << PrintExpr(op->extents[1])
    << "), " << PrintExpr(op->extents[2]) << "], dtype=" << dtype_str
    << ", buffer=" << GetStorageScopeStr(scope) << ")\n";
  } else {
    stream << vid << " = nl.ndarray(shape=" << op->extents << ", dtype=" << dtype_str
         << ", buffer=" << GetStorageScopeStr(scope) << ")\n";
  }
  this->PrintStmt(op->body);
}

void CodeGenTrainium::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::tensorized_nki_instruction) {
    ctx_.tensorizing = true;
  }
  this->PrintStmt(op->body);
  if (op->attr_key == tir::attr::tensorized_nki_instruction) {
    ctx_.tensorizing = false;
  }
}



void CodeGenTrainium::VisitStmt_(const ForNode* op) {
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  ICHECK(is_zero(op->min));
  if (ctx_.tensorizing) {
    stream << vid << " = nl.arange(" << extent << ")\n";
    ctx_.tensorized_loop_vars.insert(op->loop_var.get());
    PrintStmt(op->body);
    ctx_.tensorized_loop_vars.erase(op->loop_var.get());
  } else {
    stream << "for "<< vid << " in nl.sequential_range(" << extent << "):\n";
    int for_scope = BeginScope();
    PrintStmt(op->body);
    EndScope(for_scope);
  }
}

std::string CodeGenTrainium::PrintIndices(const Array<PrimExpr>& indices) {
  std::ostringstream os;
  ctx_.buffer_index = 0;
  ctx_.used_var_cnt = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    PreOrderVisit(indices[i], [&](const ObjectRef& node) {
      if (const auto* v = node.as<VarNode>()) {
        if (ctx_.tensorized_loop_vars.count(v)) {
          ctx_.used_var_cnt++;
        }
      }
      return true;
    });
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << PrintExpr(indices[i]);
  }
  ctx_.buffer_index = -1;
  return os.str();
}

void CodeGenTrainium::VisitStmt_(const BufferStoreNode* op) {
  LOG(FATAL) << "Trainium codegen does not support buffer store";
}

void CodeGenTrainium::VisitStmt_(const EvaluateNode* op) {
  if (is_const_int(op->value)) return;
  std::string vid = this->PrintExpr(op->value);
  if (vid != "") {
    this->PrintIndent();
    this->stream << vid << "\n";
  }
}

void CodeGenTrainium::VisitExpr_(const BufferLoadNode* op, std::ostream& os) {
  std::string buffer_str;
  if (buffer_idmap_.count(op->buffer)) {
    buffer_str = buffer_idmap_[op->buffer];
  } else {
    buffer_str = GetVarID(op->buffer->data.get());
  }
  os << buffer_str << "[";
  os << PrintIndices(op->indices);
  os << "]";
}


void CodeGenTrainium::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  CHECK(!op->op.as<GlobalVarNode>())
      << "CodegenTrainium does not support inter-function calls, "
      << "but expression " << GetRef<Call>(op) << " calls PrimFunc " << op->op;
  if (op->op.same_as(builtin::nki_matmul())) {
    ICHECK_EQ(op->args.size(), 4);
    std::string accum = is_one(op->args[3])? " += ":" = ";
    os << PrintExpr(op->args[0]) << accum << "nisa.nc_matmul(" << PrintExpr(op->args[1]) << "," <<
        PrintExpr(op->args[2]) << ")";
  } else if (op->op.same_as(builtin::nki_load())){
    ICHECK_EQ(op->args.size(), 2);
    os << PrintExpr(op->args[0]) << " = nl.load(" << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(builtin::nki_store())){
    ICHECK_EQ(op->args.size(), 2);
    os <<  "nl.store(" <<PrintExpr(op->args[0]) <<", "<< PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(builtin::nki_tensor_copy())){
    ICHECK_EQ(op->args.size(), 2);
    os << PrintExpr(op->args[0]) << " = nisa.tensor_copy(" << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(builtin::nki_activation())){
    ICHECK_EQ(op->args.size(), 5);
    // nki_activation(result, data, opcode, bias, scale)
    ICHECK(opcode_map_.count(op->args[2].as<StringImmNode>()->value));
    std::string nki_op = opcode_map_[op->args[2].as<StringImmNode>()->value];
    os << PrintExpr(op->args[0]) << " = nisa.activation(op=" << nki_op << ", data=" << PrintExpr(op->args[2]) << ",";
    os << "bias=" << PrintExpr(op->args[3]) << ", scale=" << PrintExpr(op->args[4]) << ")";
  } else if (op->op.same_as(builtin::nki_reciprocal())){
    ICHECK_EQ(op->args.size(), 2);
    os << PrintExpr(op->args[0]) << " = nisa.reciprocal(" << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(builtin::nki_tensortensor())){
    ICHECK_EQ(op->args.size(), 4);
    // nki_tensortensor(result, data1, data2, opcode)
    ICHECK(opcode_map_.count(op->args[3].as<StringImmNode>()->value));
    std::string nki_op = opcode_map_[op->args[3].as<StringImmNode>()->value];
    os << PrintExpr(op->args[0]) << " = nisa.tensor_tensor(" << PrintExpr(op->args[1]) << ", ";
    os << PrintExpr(op->args[2]) << ", op=" << nki_op << ")";
  } else if (op->op.same_as(builtin::nki_tensorscalar())){
    ICHECK_EQ(op->args.size(), 5);
    // nki_tensorscalar(result, operand1, operand2, opcode, reorder)
    ICHECK(opcode_map_.count(op->args[3].as<StringImmNode>()->value));
    std::string nki_op = opcode_map_[op->args[3].as<StringImmNode>()->value];
    int reorder = op->args[4].as<IntImmNode>()->value;
    bool is_reorder = reorder != 0;
    os << PrintExpr(op->args[0]) << " = nisa.tensor_scalar(" << PrintExpr(op->args[1]) << ", operand0=";
    os << PrintExpr(op->args[2]) << ", op0=" << nki_op << ", reverse0=" << is_reorder << ")";
  } else if (op->op.same_as(builtin::nki_memset())){
    ICHECK_GE(op->args.size(), 2);
    // result, value
    os << PrintExpr(op->args[0]) << " = " << PrintExpr(op->args[1]);
  } else if (op->op.same_as(builtin::nki_transpose())){
    ICHECK_EQ(op->args.size(), 2);
    os << PrintExpr(op->args[0]) << " = nl.transpose(" << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(builtin::nki_sum())){
    ICHECK_EQ(op->args.size(), 3);
    os << PrintExpr(op->args[0]) << " = nl.sum(" << PrintExpr(op->args[1]) << ", axis=" << PrintExpr(op->args[2]) << ")";
  } else {
    LOG(FATAL)<< "Trainium codegen does not support call to " << op->op;
  }
}

void CodeGenTrainium::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  std::ostringstream temp;
  if (std::isinf(op->value)) {
    if (op->value < 0) {
      temp << "-";
    }
    temp << "math.inf";
  } else if (std::isnan(op->value)) {
    LOG(FATAL)<< "Trainium codegen does not support NaN";
  } else {
    temp << std::scientific << op->value;
  }
  MarkConst(temp.str());
  os << temp.str();
}

void CodeGenTrainium::VisitExpr_(const VarNode* op, std::ostream& os) {  // NOLINT(*)
  os << GetVarID(op);
  if(ctx_.buffer_index == -1){
    return;
  }
  if(!ctx_.tensorized_loop_vars.count(op)){
    return;
  }
  os << "[";
  for (int i = 0; i < static_cast<int>(ctx_.used_var_cnt); i++) {
    if(i == ctx_.buffer_index){
      os << ":, ";
    } else {
      os << "None, ";
    }
  }
  os << "]";
  ctx_.buffer_index++;
}

void CodeGenTrainium::VisitExpr_(const CastNode* op, std::ostream& os){
  ctx_.dst_dtype = op->dtype;
  CodeGenTrainium::VisitExpr(op->value, os);
}

void CodeGenTrainium::VisitExpr_(const FloorDivNode* op, std::ostream& os) {
  os << PrintExpr(op->a) << " // " << PrintExpr(op->b);
}

void CodeGenTrainium::VisitExpr_(const FloorModNode* op, std::ostream& os) {
  os << PrintExpr(op->a) << " % " << PrintExpr(op->b);
}

void CodeGenTrainium::VisitStmt_(const DeclBufferNode* op) {
  if(op->buffer.scope() == "trn.psum" || op->buffer.scope() == "trn.sbuf"){
    PrintStmt(op->body);
    return;
  }
  std::string data_vid = GetVarID(op->buffer->data.get());
  std::string buffer_vid = name_supply_->FreshName(data_vid + "_buffer");
  buffer_idmap_[op->buffer] = buffer_vid;
  PrintIndent();
  stream << buffer_vid << " = " << data_vid << ".reshape(" << op->buffer->shape << ")\n";
  PrintStmt(op->body);
}

runtime::Module BuildTrainium(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;

  std::ostringstream source_maker;
  std::unordered_map<std::string, std::string> smap;
  const auto* fTrainium_compile = Registry::Get("tvm_callback_Trainium_compile");
  std::string fmt = fTrainium_compile ? "Trainiumlib" : "Trainium";

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenTrainium: Can only take PrimFunc";
    auto global_symbol = kv.second->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(global_symbol.defined());
    std::string func_name = global_symbol.value();
    source_maker << "# Function: " << func_name << "\n";
    CodeGenTrainium cg(target);
    cg.Init(output_ssa);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(kv.first, f);

    std::string fsource = cg.Finish();
    source_maker << fsource << "\n";
    smap[func_name] = fsource;
  }

  return codegen::DeviceSourceModuleCreate(source_maker.str(), fmt, ExtractFuncInfo(mod), "nki");
}

TVM_REGISTER_GLOBAL("target.build.trn").set_body_typed(BuildTrainium);
}  // namespace codegen
}  // namespace tvm
