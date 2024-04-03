

# Original file: ./hf_BigBird___60.0/hf_BigBird___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/y6/cy6lel34xnadbzy3em6uoq5wycum46blfd6d7o4mbwwbr7m7kjhg.py
# Source Nodes: [cat_303, cat_304, cat_310, cat_311], Original ATen: [aten.cat]
# cat_303 => cat_10
# cat_304 => cat_9
# cat_310 => cat_3
# cat_311 => cat_2
triton_poi_fused_cat_9 = async_compile.triton('triton_poi_fused_cat_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 12288)
    x3 = xindex % 12288
    x0 = xindex % 64
    x1 = (xindex // 64) % 192
    tmp0 = tl.load(in_ptr0 + ((186*x2) + (x3 // 4096)), None)
    tmp6 = tl.load(in_ptr0 + (183 + (186*x2) + (186*((183 + (x3 // 4096)) // 186)) + (186*((749568 + x3) // 761856)) + (x3 // 4096)), None)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = 64*x2
    tmp3 = tmp1 + tmp2
    tmp4 = tl.where(tmp3 < 0, tmp3 + 768, tmp3)
    # tl.device_assert((0 <= tmp4) & (tmp4 < 768), "index out of bounds: 0 <= tmp4 < 768")
    tmp5 = tl.load(in_ptr1 + (x0 + (64*((tmp4 // 64) % 12)) + (768*(x1 % 64)) + (49152*(tmp4 % 64))), None)
    tmp7 = tmp6.to(tl.int64)
    tmp8 = (64*x2) + (64*((183 + (x3 // 4096)) // 186)) + (64*((749568 + x3) // 761856))
    tmp9 = tmp7 + tmp8
    tmp10 = tl.where(tmp9 < 0, tmp9 + 768, tmp9)
    # tl.device_assert((0 <= tmp10) & (tmp10 < 768), "index out of bounds: 0 <= tmp10 < 768")
    tmp11 = tl.load(in_ptr1 + (x0 + (64*((tmp10 // 64) % 12)) + (768*(x1 % 64)) + (49152*(tmp10 % 64))), None)
    tmp12 = tl.load(in_ptr2 + (x0 + (64*((tmp4 // 64) % 12)) + (768*(x1 % 64)) + (49152*(tmp4 % 64))), None)
    tmp13 = tl.load(in_ptr2 + (x0 + (64*((tmp10 // 64) % 12)) + (768*(x1 % 64)) + (49152*(tmp10 % 64))), None)
    tl.store(out_ptr0 + (x3 + (28672*x2)), tmp5, None)
    tl.store(out_ptr1 + (x3 + (28672*x2)), tmp11, None)
    tl.store(out_ptr2 + (x3 + (28672*x2)), tmp12, None)
    tl.store(out_ptr3 + (x3 + (28672*x2)), tmp13, None)
''')
