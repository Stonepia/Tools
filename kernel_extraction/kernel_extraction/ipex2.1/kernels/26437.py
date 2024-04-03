

# Original file: ./vision_maskrcnn__25_inference_65.5/vision_maskrcnn__25_inference_65.5_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/oa/coaybv7ze3ynw57bql2joibsm3aolvbsmdv5perju7htvr3yxasd.py
# Source Nodes: [stack_5], Original ATen: [aten.stack]
# stack_5 => cat_8
triton_poi_fused_stack_31 = async_compile.triton('triton_poi_fused_stack_31', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_31', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_stack_31(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 242991
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + (4*x0)), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (3 + (4*x0)), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (1 + (4*x0)), xmask).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (3 + (4*x0)), xmask).to(tl.float32)
    tmp1 = 1.0
    tmp2 = tmp0 / tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = tmp2 * tmp5
    tmp7 = 0.5
    tmp8 = tmp5 * tmp7
    tmp9 = tmp4 + tmp8
    tmp10 = tmp6 + tmp9
    tmp12 = tmp11 / tmp1
    tmp13 = tmp12.to(tl.float32)
    tmp14 = 4.135166556742356
    tmp15 = triton_helpers.minimum(tmp13, tmp14)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.exp(tmp16)
    tmp18 = tmp17 * tmp5
    tmp19 = tmp7 * tmp18
    tmp20 = tmp10 - tmp19
    tmp21 = tmp10 + tmp19
    tl.store(out_ptr0 + (4*x0), tmp20, xmask)
    tl.store(out_ptr1 + (4*x0), tmp21, xmask)
''')
