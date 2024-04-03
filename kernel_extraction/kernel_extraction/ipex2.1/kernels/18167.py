

# Original file: ./pnasnet5large___60.0/pnasnet5large___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/oj/coj6xvxqypgorx77r23fjs5z6gzhgproljoikt4ug5kfh2qxdwfa.py
# Source Nodes: [pad], Original ATen: [aten.constant_pad_nd]
# pad => constant_pad_nd
triton_poi_fused_constant_pad_nd_10 = async_compile.triton('triton_poi_fused_constant_pad_nd_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43869696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 16224) % 169
    x1 = (xindex // 96) % 169
    x3 = (xindex // 2741856)
    x4 = xindex % 16224
    x6 = xindex
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 165, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-31872) + x4 + (15840*x2) + (2613600*x3)), tmp10 & xmask, other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tl.store(out_ptr0 + (x6), tmp12, xmask)
''')