

# Original file: ./tnt_s_patch16_224___60.0/tnt_s_patch16_224___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/kg/ckg7q7sjtgjf6lctw5xprjgx3lynwc3j3il63guaqvap5xfz7h2y.py
# Source Nodes: [add_4], Original ATen: [aten.add]
# add_4 => add_17
triton_poi_fused_add_14 = async_compile.triton('triton_poi_fused_add_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*bf16', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_14(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9633792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 75264
    x1 = (xindex // 75264)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (384 + x0 + (75648*x1)), None)
    tmp1 = tl.load(in_ptr1 + (384 + x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tl.store(out_ptr0 + (x0 + (75648*x1)), tmp5, None)
''')
