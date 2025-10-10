# 1. Goal and Vision
The primary goal of this framework is to enable the implementation of kernels with minimal programming effort while achieving state-of-the-art performance. The system must be flexible, allowing developers to easily navigate the trade-offs between different programming models—from high-level, automated approaches to fine-grained, expert-driven control. Specifically, we want to hightlight kernels for Large Language Model (LLM) workloads, which are the most important kernels in the current, and possibly near future AI landscape.

The vision is a holistic infrastructure where developers can express computational intent at the most appropriate level of abstraction. Whether it's by composing pre-built library functions, using a productive Tile-based DSL, or even leveraging AI to generate kernels, the framework should provide the necessary tools to compile and optimize for target hardware efficiently.

# 2. Target Kernels in LLM Workloads
The framework is designed to target general GPU kernel programming. But we specifically focus on critical kernels that are computational bottlenecks in modern LLM inference and training. Key examples include:

- **Flashinfer Kernels (Attention):** The attention mechanism is the core of the Transformer architecture. Kernels like Flashinfer and FlashAttention are crucial for optimizing this mechanism by avoiding the materialization of the large N x N attention matrix in global GPU memory. They use tiling, recomputation, and careful management of on-chip SRAM to achieve significant speedups and memory savings, which is essential for handling long sequences.

- **GeMM (General Matrix-Matrix Multiplication), Grouped GeMM:** LLMs are fundamentally composed of massive linear and feed-forward layers, which are executed as GeMM operations. Performance of the entire model is heavily dependent on the efficiency of these kernels. Optimizations focus on maximizing the utilization of specialized hardware units like NVIDIA's Tensor Cores, managing data layouts, and tuning for specific matrix sizes and data types (e.g., FP16, INT8, FP8).

- **Layernorm / RMSNorm:** These normalization kernels are applied repeatedly throughout the model architecture. While not as computationally dominant as GeMM or Attention, they can become a bottleneck due to their memory-bound nature (they perform few computations per byte of data read). Efficient implementations fuse operations, maximize memory bandwidth, and parallelize effectively across the input tensors.

- **Communication (possibly fused with computation) Kernels:** Communication is a critical bottleneck in distributed training. Communication kernels are crucial for optimizing the communication between different devices.

# 3. Programming Models & Abstraction Layers
The framework provides a spectrum of programming abstractions, allowing the user to choose the right tool for the job. These layers are not mutually exclusive; higher layers are typically built upon and compile down to lower layers.

##  3.1. Supporting Infrastructure
A robust set of infrastructure components underpins the entire framework. Some of the key components are:

### FFI (Foreign Function Interface)
A crucial component that allows multiple languages to interact with each other. A Packed Function system is often used to pass typed objects and function calls across the language boundary in a standardized way. For example, the Python frontend can call C++ backend, and vice versa.

Not only we use FFI to support the compiled artifacts and runtime services, but also to build the compiler itself.

### Python Parser/Printer
This component is responsible for parsing the user's Python DSL code into the framework's internal IR and, conversely, printing the IR back out as human-readable code for debugging.

### Runtime
Provides essential runtime services, including the management of fundamental dtypes (e.g., float32, bfloat16), device APIs for memory allocation, and kernel launching.

### Distributed (e.g., Disco)
To scale to multi-GPU and multi-node systems, a distributed runtime is needed. It handles communication primitives (e.g., AllReduce), manages device topology, and orchestrates distributed computations.

### Transformation Utils (Visitors/Mutators)
The compiler is built on a set of utilities for program analysis and transformation. The Visitor pattern is used to traverse the IR to gather information, while the Mutator pattern is used to traverse and rewrite the IR to apply optimizations.

### RPC (Remote Procedure Call)
Used by the distributed runtime to allow different processes, potentially on different machines, to communicate and coordinate with each other.

### Arith
A specialized library for analyzing and simplifying arithmetic expressions.

## 3.2. Hardware Abstraction (CUDA/PTX, NKI)
This is the lowest level of the software stack and our lanauge design, representing the interface with the hardware. It aims to faithfully represent the hardware and the native programming model (such as CUDA/PTX, NKI), and provide a foundation for the higher-level abstractions.

### Native Ops (e.g., mma.sync, ldmatrix)
This refers to the instruction set for the hardware. PTX (Parallel Thread Execution) is NVIDIA's virtual instruction set architecture (ISA). Code is typically generated in PTX, and the GPU driver performs the final just-in-time compilation to the native machine code (SASS). NKI (Neuron Kernel Interface) is an analogous ISA for AWS Trainium/Inferentia accelerators.

### Buffer
At this level, a buffer is just a region of memory in a specific state space (e.g., global, shared, register). All higher-level Tensor abstractions ultimately resolve to operations on these raw buffers.

Note that buffers can natively be 2-dimensional (such as Tensor Memory on Blackwell architectures and SRAM on Trainium) or even higher-dimensional due to the evolution of the hardware.

### Primitive Expressions/Statements
The code is represented as a low-level IR consisting of basic statements (For, While, IfThenElse) and expressions (Add, Sub, Mul). This is the form upon which code generation (Codegen) operates.

### Codegen
The final stage of the compiler that traverses the low-level IR and emits source code in a target language like PTX or CUDA C++.

## 3.3. Libraries (CUTLASS, CuTe, CuB, ThunderKitten)
To gain maximal control and to access the most optimized routines, the framework wraps the routine implementations in a unified interface called libraries. These libraries provide a vocabulary of highly-tuned, reusable components for building kernels. DSLs often use these libraries as compilation targets.

### Exec Scope
For SIMT architectures, we need to provide abstractions to explicitly manage the GPU's execution hierarchy: threads, warps, thread blocks, and clusters. The routines in the libraries require the knowledge of available execution hierarchy to dispatch the appropriate implementation.

### Layout
A critical concept in performance programming. Layouts of input/output tensors can largely determine the optimization decisions of routine implementations.

### Tensors
They provide powerful Tensor objects that bundle a pointer to a buffer with a Layout. This enables developers to write logical, coordinate-based code while the library handles the complex address calculations.

### Ops & Dispatches
Rountines are represented as ops. These libraries contain collections of highly optimized ops (e.g., gemm, copy_async). A dispatch mechanism selects the best implementation (or "kernel schedule") based on the problem size, data type, and target hardware architecture.

### Asynchronous Ops
To hide the high latency of global memory access, modern GPUs support asynchronous copy operations (e.g., TMA Load on Hopper). We need to provide abstractions to schedule data movement from global to shared memory concurrently with computation on a previous set of data, creating a software pipeline that keeps the compute units constantly fed. This is a key optimization for the performance of the kernels. Considering the increasing amount of asynchronous operations (copy, GeMM, etc.) along with their completion mechanisms in the kernels, we intend to have a generic abstration.

## 3.4. Tile DSLs (Triton, CuTile, TileLang)
This is the primary interface for productive kernel development. A Tile DSL exposes a Python-based, domain-specific language that allows developers to think in terms of "tiles" or blocks of data, which maps naturally to the hierarchical memory and execution model of GPUs. Triton is the most popular and successful tile DSL for GPU kernel development. Post its popularity, many other tile DSLs have been proposed with the motivation that Triton lacks some sort of finer-grained control and proposes solutions that tweaks Triton's programming model and transformations to accomodate the needs of the users.

### High-level DSL
The DSL provides a simplified view of the hardware. The programmer writes a kernel that describes the computation for a single program/work-group, and the DSL compiler handles the complex mapping to the full grid of thread blocks. It usually abstracts away explicit synchronization, address arithmetic, tensor storage, and other low-level details.

### Transformations
The power of the DSL comes from its compiler's ability to apply a series of performance-oriented transformations. The user can often provide hints or directives (e.g., tile sizes, pipeline stages) to guide the compiler. The compiler then lowers this high-level representation into an intermediate representation (IR) where it can apply optimizations like loop unrolling, instruction scheduling, and memory access coalescing.
