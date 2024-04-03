

# Original file: ./MobileBertForMaskedLM__0_forward_280.0/MobileBertForMaskedLM__0_forward_280.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/md/cmdnnfce3kuxtvm2pqg6pfo2ayd2a33j6eh7cokogb7p5rhw3gb4.py
# Source Nodes: [pad, pad_1], Original ATen: [aten.constant_pad_nd]
# pad => constant_pad_nd
# pad_1 => constant_pad_nd_1
triton_poi_fused_constant_pad_nd_1 = async_compile.triton('triton_poi_fused_constant_pad_nd_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128) % 128
    x3 = (xindex // 128)
    x0 = xindex % 128
    tmp0 = x1
    tmp1 = tl.full([1], 127, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (1 + x3), tmp2, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.where(tmp3 < 0, tmp3 + 30522, tmp3)
    # tl.device_assert(((0 <= tmp4) & (tmp4 < 30522)) | ~tmp2, "index out of bounds: 0 <= tmp4 < 30522")
    tmp5 = tl.load(in_ptr1 + (x0 + (128*tmp4)), tmp2, other=0.0)
    tmp6 = tl.where(tmp2, tmp5, 0.0)
    tmp7 = (-1) + x1
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tl.load(in_ptr0 + ((-1) + x3), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.where(tmp10 < 0, tmp10 + 30522, tmp10)
    # tl.device_assert(((0 <= tmp11) & (tmp11 < 30522)) | ~tmp9, "index out of bounds: 0 <= tmp11 < 30522")
    tmp12 = tl.load(in_ptr1 + (x0 + (128*tmp11)), tmp9, other=0.0)
    tmp13 = tl.where(tmp9, tmp12, 0.0)
    tl.store(out_ptr0 + (x0 + (384*x3)), tmp6, None)
    tl.store(out_ptr1 + (x0 + (384*x3)), tmp13, None)
''')
