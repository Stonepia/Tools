

# Original file: ./crossvit_9_240___60.0/crossvit_9_240___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/mg/cmguxucwwkzqnfciwl3qe4dwvhogyjhqnpxyklvparlnlqdn2gmt.py
# Source Nodes: [cat_23, cat_24], Original ATen: [aten.cat]
# cat_23 => cat_4
# cat_24 => cat_3
triton_poi_fused_cat_40 = async_compile.triton('triton_poi_fused_cat_40', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_40', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6553600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 51200
    x1 = (xindex // 51200)
    tmp0 = tl.load(in_ptr0 + (128 + x0 + (51328*x1)), None)
    tmp1 = tl.load(in_ptr1 + (128 + x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (128 + x0 + (51328*x1)), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (128 + x0 + (51328*x1)), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 + tmp7
    tl.store(out_ptr0 + (x0 + (51328*x1)), tmp8, None)
    tl.store(out_ptr1 + (x0 + (51328*x1)), tmp8, None)
''')