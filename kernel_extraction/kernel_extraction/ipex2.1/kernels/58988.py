

# Original file: ./AllenaiLongformerBase__22_backward_143.5/AllenaiLongformerBase__22_backward_143.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/47/c473lplip747cksslpc2hk3mw3zb26glh5krlmbufq5hvrzwf7j7.py
# Source Nodes: [], Original ATen: [aten.index_add, aten.new_zeros]

triton_poi_fused_index_add_new_zeros_12 = async_compile.triton('triton_poi_fused_index_add_new_zeros_12', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_add_new_zeros_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_index_add_new_zeros_12(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9437184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + ((16384*((x0 // 49152) % 4)) + (98304*(x0 // 196608)) + (x0 % 49152)), None)
    tmp2 = tl.load(in_ptr1 + (x0), None)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 4718592, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 4718592), "index out of bounds: 0 <= tmp1 < 4718592")
    tl.atomic_add(out_ptr0 + (tl.broadcast_to(tmp1, [XBLOCK])), tmp2, None)
''')
