

# Original file: ./hf_Longformer__22_inference_62.2/hf_Longformer__22_inference_62.2_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/h6/ch6f46eytk7xrs7btcz6ai7hg6dwhkwabqgvcltllanc7moxl75o.py
# Source Nodes: [pad_3], Original ATen: [aten.constant_pad_nd]
# pad_3 => constant_pad_nd_3
triton_poi_fused_constant_pad_nd_17 = async_compile.triton('triton_poi_fused_constant_pad_nd_17', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37847040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 770
    x2 = (xindex // 9240)
    x3 = (xindex // 770)
    x1 = (xindex // 770) % 12
    tmp0 = x0
    tmp1 = tl.full([1], 513, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x2), tmp2, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0 + (513*x3)), tmp2, other=0.0)
    tmp5 = tl.load(in_ptr2 + (x3), tmp2, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp4 - tmp5
    tmp7 = tl.exp(tmp6)
    tmp8 = tl.load(in_ptr3 + (x3), tmp2, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 / tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp3, tmp10, tmp9)
    tmp12 = tl.where(tmp2, tmp11, 0.0)
    tl.store(out_ptr0 + (x0 + (770*x2) + (3153920*x1)), tmp12, None)
''')
