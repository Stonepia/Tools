

# Original file: ./levit_128___60.0/levit_128___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/ya/cyaopsbz3yqvnxs5mray5kxoyprjieulmhs4vx7wjuwxnyipa3jw.py
# Source Nodes: [add_39, l__mod___head_bn, l__mod___head_dist_bn, mean], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean]
# add_39 => add_194
# l__mod___head_bn => add_196, convert_element_type_278, mul_232, mul_233, sub_76
# l__mod___head_dist_bn => add_198, convert_element_type_281, mul_235, mul_236, sub_77
# mean => mean
triton_per_fused__native_batch_norm_legit_no_training_add_mean_40 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_mean_40', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: '*fp32', 8: '*bf16', 9: '*bf16', 10: '*bf16', 11: '*bf16', 12: '*bf16', 13: '*bf16', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_40', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_mean_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (6144*x1)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (384*r2) + (6144*x1)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp24 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp31 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp35 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp38 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 * tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp0 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = 16.0
    tmp21 = tmp19 / tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp25 = tmp23 - tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 * tmp29
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp30 + tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp27 * tmp36
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp37 + tmp39
    tmp41 = tmp40.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp34, None)
    tl.store(out_ptr2 + (x3), tmp41, None)
''')
