

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/w2/cw2fr5bes6sbivrew7i2o5ssxv7firasz67diqzd3husmdijr6kc.py
# Source Nodes: [grid_sample_5], Original ATen: [aten.grid_sampler_2d]
# grid_sample_5 => add_131, add_132, index_28, index_29, index_30, mul_231, mul_232, mul_233
triton_poi_fused_grid_sampler_2d_52 = async_compile.triton('triton_poi_fused_grid_sampler_2d_52', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*i64', 6: '*fp32', 7: '*i64', 8: '*i64', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_52', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton_poi_fused_grid_sampler_2d_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2230272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 123904
    x2 = (xindex // 371712)
    x3 = (xindex // 123904)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (123904*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0 + (123904*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0 + (123904*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0 + (123904*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0 + (123904*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x0 + (123904*x2)), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (x0 + (123904*x2)), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr8 + (x0 + (123904*x2)), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr9 + (x0 + (123904*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.where(tmp0 < 0, tmp0 + 352, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 352), "index out of bounds: 0 <= tmp1 < 352")
    tmp3 = tl.where(tmp2 < 0, tmp2 + 352, tmp2)
    # tl.device_assert((0 <= tmp3) & (tmp3 < 352), "index out of bounds: 0 <= tmp3 < 352")
    tmp4 = tl.load(in_ptr2 + (tmp3 + (352*tmp1) + (123904*x3)), None)
    tmp6 = tmp4 * tmp5
    tmp8 = tl.where(tmp7 < 0, tmp7 + 352, tmp7)
    # tl.device_assert((0 <= tmp8) & (tmp8 < 352), "index out of bounds: 0 <= tmp8 < 352")
    tmp10 = tl.where(tmp9 < 0, tmp9 + 352, tmp9)
    # tl.device_assert((0 <= tmp10) & (tmp10 < 352), "index out of bounds: 0 <= tmp10 < 352")
    tmp11 = tl.load(in_ptr2 + (tmp10 + (352*tmp8) + (123904*x3)), None)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp6 + tmp13
    tmp16 = tl.where(tmp15 < 0, tmp15 + 352, tmp15)
    # tl.device_assert((0 <= tmp16) & (tmp16 < 352), "index out of bounds: 0 <= tmp16 < 352")
    tmp18 = tl.where(tmp17 < 0, tmp17 + 352, tmp17)
    # tl.device_assert((0 <= tmp18) & (tmp18 < 352), "index out of bounds: 0 <= tmp18 < 352")
    tmp19 = tl.load(in_ptr2 + (tmp18 + (352*tmp16) + (123904*x3)), None)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp14 + tmp21
    tl.store(out_ptr0 + (x4), tmp22, None)
''')
