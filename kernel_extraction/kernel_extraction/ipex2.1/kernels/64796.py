

# Original file: ./basic_gnn_gcn__25_inference_65.5/basic_gnn_gcn__25_inference_65.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/5l/c5l4bsno2cf6h3f5ojsoxd62t7hwyd6ltofuhpelgucivr36npot.py
# Source Nodes: [index_select, mul, new_zeros, scatter_add_], Original ATen: [aten.index_select, aten.mul, aten.new_zeros, aten.scatter_add]
# index_select => index
# mul => mul
# new_zeros => full_default
# scatter_add_ => scatter_add
triton_poi_fused_index_select_mul_new_zeros_scatter_add_1 = async_compile.triton('triton_poi_fused_index_select_mul_new_zeros_scatter_add_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_index_select_mul_new_zeros_scatter_add_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 640000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')
