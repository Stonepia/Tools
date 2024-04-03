

# Original file: ./timm_resnest___60.0/timm_resnest___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/g7/cg7flwi665f5kny34bw6iebyreoxzkhct34xb6una37wqnu2cqkk.py
# Source Nodes: [mul_1, sum_4], Original ATen: [aten.mul, aten.sum]
# mul_1 => mul_34
# sum_4 => sum_6
triton_poi_fused_mul_sum_6 = async_compile.triton('triton_poi_fused_mul_sum_6', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_mul_sum_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x3 = (xindex // 128)
    x2 = (xindex // 401408)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x3)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (128 + x0 + (256*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (128 + x0 + (256*x3)), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = triton_helpers.maximum(tmp2, tmp4)
    tmp6 = tmp2 - tmp5
    tmp7 = tl.exp(tmp6)
    tmp8 = tmp4 - tmp5
    tmp9 = tl.exp(tmp8)
    tmp10 = tmp7 + tmp9
    tmp11 = tmp7 / tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp0 * tmp12
    tmp15 = tmp9 / tmp10
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp18 = tmp13 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''')
