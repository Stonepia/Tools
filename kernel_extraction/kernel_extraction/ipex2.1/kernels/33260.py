

# Original file: ./hf_Whisper___60.0/hf_Whisper___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/bi/cbigyzwvkxppjexcm4iu4vpweaiveavhjlzrbv74fuuvxppmpv2d.py
# Source Nodes: [add, l__mod___encoder_layers_0_self_attn_layer_norm], Original ATen: [aten.add, aten.native_layer_norm]
# add => add_2
# l__mod___encoder_layers_0_self_attn_layer_norm => add_3, add_4, clone_1, mul_6, mul_7, rsqrt, sub, var_mean
triton_red_fused_add_native_layer_norm_1 = async_compile.triton('triton_red_fused_add_native_layer_norm_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12000
    rnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1500
    x1 = (xindex // 1500)
    tmp14_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp14_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp14_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1500*r2) + (576000*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = 0.5
        tmp4 = tmp2 * tmp3
        tmp5 = 0.7071067811865476
        tmp6 = tmp2 * tmp5
        tmp7 = libdevice.erf(tmp6)
        tmp8 = 1.0
        tmp9 = tmp7 + tmp8
        tmp10 = tmp4 * tmp9
        tmp12 = tmp10 + tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp14_mean_next, tmp14_m2_next, tmp14_weight_next = triton_helpers.welford_reduce(
            tmp13, tmp14_mean, tmp14_m2, tmp14_weight,
        )
        tmp14_mean = tl.where(rmask & xmask, tmp14_mean_next, tmp14_mean)
        tmp14_m2 = tl.where(rmask & xmask, tmp14_m2_next, tmp14_m2)
        tmp14_weight = tl.where(rmask & xmask, tmp14_weight_next, tmp14_weight)
    tmp14_tmp, tmp15_tmp, tmp16_tmp = triton_helpers.welford(
        tmp14_mean, tmp14_m2, tmp14_weight, 1
    )
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp17 = tl.load(in_ptr0 + (x0 + (1500*r2) + (576000*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr2 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tmp17 + tmp18
        tmp20 = 0.5
        tmp21 = tmp19 * tmp20
        tmp22 = 0.7071067811865476
        tmp23 = tmp19 * tmp22
        tmp24 = libdevice.erf(tmp23)
        tmp25 = 1.0
        tmp26 = tmp24 + tmp25
        tmp27 = tmp21 * tmp26
        tmp29 = tmp27 + tmp28
        tmp30 = tmp29 - tmp14
        tmp31 = 384.0
        tmp32 = tmp15 / tmp31
        tmp33 = 1e-05
        tmp34 = tmp32 + tmp33
        tmp35 = libdevice.rsqrt(tmp34)
        tmp36 = tmp30 * tmp35
        tmp38 = tmp36 * tmp37
        tmp39 = tmp38 + tmp18
        tl.store(out_ptr2 + (r2 + (384*x3)), tmp39, rmask & xmask)
''')
