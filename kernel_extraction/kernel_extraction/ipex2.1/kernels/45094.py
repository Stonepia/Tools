

# Original file: ./eca_halonext26ts___60.0/eca_halonext26ts___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/3l/c3lscfslgg7cijk6tvjjnb4rgne5oh7p7irko7prm53lwdmptw5s.py
# Source Nodes: [matmul_3], Original ATen: [aten.clone]
# matmul_3 => clone_7
triton_poi_fused_clone_28 = async_compile.triton('triton_poi_fused_clone_28', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_28', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18874368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 32) % 144
    x2 = (xindex // 4608) % 4
    x0 = xindex % 32
    x3 = (xindex // 18432)
    x4 = xindex
    tmp0 = (-2) + (8*((x1 + (144*x2)) // 288)) + (x1 // 12)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + (8*(x2 % 2)) + (x1 % 12)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-13056) + (384*(x1 % 12)) + (3072*(x2 % 2)) + (6144*(x1 // 12)) + (49152*((x1 + (144*x2)) // 288)) + (98304*((9216 + x1 + (144*x2) + (576*x0) + (27648*x3)) // 221184)) + (((9216 + x1 + (144*x2) + (576*x0) + (27648*x3)) // 576) % 384)), tmp10, other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tl.store(out_ptr0 + (x4), tmp12, None)
''')
