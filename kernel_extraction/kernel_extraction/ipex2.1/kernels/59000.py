

# Original file: ./AllenaiLongformerBase__22_backward_143.5/AllenaiLongformerBase__22_backward_143.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/ve/cvej76zxvgt57cr2hztaxzl2e4wvkahkg3it7yt2ka6ploy4qscu.py
# Source Nodes: [], Original ATen: [aten.index_add, aten.new_zeros]

triton_poi_fused_index_add_new_zeros_24 = async_compile.triton('triton_poi_fused_index_add_new_zeros_24', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0', 'out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_add_new_zeros_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_index_add_new_zeros_24(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x0 // 98304)) + (3072*((x0 // 64) % 512)) + (786432*((x0 // 32768) % 3)) + (x0 % 64)), None)
    tmp2 = tl.load(in_ptr1 + ((512*(x0 % 64)) + (32768*(x0 // 32768)) + ((x0 // 64) % 512)), None)
    tmp3 = tl.load(in_ptr2 + (x0), None)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 3145728, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 3145728), "index out of bounds: 0 <= tmp1 < 3145728")
    tl.atomic_add(out_ptr0 + (tl.broadcast_to(tmp1, [XBLOCK])), tmp2, None)
    tl.atomic_add(out_ptr1 + (tl.broadcast_to(tmp1, [XBLOCK])), tmp3, None)
''')
