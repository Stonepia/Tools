

# Original file: ./hf_Longformer__22_inference_62.2/hf_Longformer__22_inference_62.2_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/wy/cwy2j3jj3zcvx3riuauxwxy6fxuywlulivmodsudwdetvo2sph6f.py
# Source Nodes: [pad_3], Original ATen: [aten.constant_pad_nd]
# pad_3 => constant_pad_nd_3
triton_poi_fused_constant_pad_nd_18 = async_compile.triton('triton_poi_fused_constant_pad_nd_18', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = tl.load(in_ptr1 + (x0 + (513*x3)), tmp2, other=0.0).to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x3), tmp2, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.load(in_ptr3 + (x3), tmp2, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 / tmp9
    tmp11 = 0.0
    tmp12 = tl.where(tmp3, tmp11, tmp10)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.where(tmp2, tmp13, 0.0)
    tl.store(out_ptr0 + (x0 + (770*x2) + (3153920*x1)), tmp14, None)
''')
