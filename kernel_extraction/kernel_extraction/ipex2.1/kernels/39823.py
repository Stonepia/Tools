

# Original file: ./levit_128___60.0/levit_128___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/fz/cfz73yeo5lhgrch4dsxziubqxfmzvpesb4kgmsijifxomahvc5m3.py
# Source Nodes: [add_39, l__mod___head_bn, l__mod___head_dist_bn, mean], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean]
# add_39 => add_194
# l__mod___head_bn => add_196, mul_232, mul_233, sub_76
# l__mod___head_dist_bn => add_198, mul_235, mul_236, sub_77
# mean => mean
triton_per_fused__native_batch_norm_legit_no_training_add_mean_38 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_mean_38', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[65536, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_38', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_mean_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 384
    x1 = (xindex // 384)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (6144*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (384*r2) + (6144*x1)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp0 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = 16.0
    tmp16 = tmp14 / tmp15
    tmp18 = tmp16 - tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr1 + (x3), tmp24, None)
    tl.store(out_ptr2 + (x3), tmp28, None)
''')
