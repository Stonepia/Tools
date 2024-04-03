

# Original file: ./nfnet_l0___60.0/nfnet_l0___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/xn/cxng65bwkunzgnhmnsiovqtkwwmta76s23sbx2pzapgh7bw5ooc6.py
# Source Nodes: [add_10, add_11, mul_101, mul_102, mul_103, mul_93, mul_94, mul_95], Original ATen: [aten.add, aten.mul]
# add_10 => add_62
# add_11 => add_67
# mul_101 => mul_264
# mul_102 => mul_265
# mul_103 => mul_266
# mul_93 => mul_244
# mul_94 => mul_245
# mul_95 => mul_246
triton_poi_fused_add_mul_31 = async_compile.triton('triton_poi_fused_add_mul_31', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_31', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15925248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 124416)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (1536*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0 + (1536*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (x3), None).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.2
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9 * tmp3
    tmp11 = tmp10 * tmp5
    tmp13 = tmp11 + tmp12
    tmp14 = tmp6 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')
