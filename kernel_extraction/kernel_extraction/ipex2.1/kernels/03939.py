

# Original file: ./basic_gnn_edgecnn___60.0/basic_gnn_edgecnn___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/vk/cvkrca6jhnwo4ofs24wzxk7u32yi3raqst2ne2y34rcmwkfyxmfr.py
# Source Nodes: [new_zeros, scatter_reduce_], Original ATen: [aten.new_zeros, aten.scatter_reduce]
# new_zeros => full_default
# scatter_reduce_ => scatter_reduce
triton_poi_fused_new_zeros_scatter_reduce_2 = async_compile.triton('triton_poi_fused_new_zeros_scatter_reduce_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_scatter_reduce_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_new_zeros_scatter_reduce_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 640000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')
