

# Original file: ./hf_Whisper___60.0/hf_Whisper___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/qv/cqvrmdyedstdw2hli4vpnroe454clkjwb3iwmgojz67peaie5fcw.py
# Source Nodes: [add, add_1, dropout_2, l__mod___encoder_layers_0_final_layer_norm], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add => add_2
# add_1 => add_5
# dropout_2 => clone_7
# l__mod___encoder_layers_0_final_layer_norm => add_6, add_7, clone_8, mul_10, mul_9, rsqrt_1, sub_2, var_mean_1
triton_red_fused_add_clone_native_layer_norm_6 = async_compile.triton('triton_red_fused_add_clone_native_layer_norm_6', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_native_layer_norm_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_clone_native_layer_norm_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12000
    rnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1500
    x1 = (xindex // 1500)
    x3 = xindex
    tmp16_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp16_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp16_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1500*r2) + (576000*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr3 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp14 = tmp12 + tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp16_mean_next, tmp16_m2_next, tmp16_weight_next = triton_helpers.welford_reduce(
            tmp15, tmp16_mean, tmp16_m2, tmp16_weight,
        )
        tmp16_mean = tl.where(rmask & xmask, tmp16_mean_next, tmp16_mean)
        tmp16_m2 = tl.where(rmask & xmask, tmp16_m2_next, tmp16_m2)
        tmp16_weight = tl.where(rmask & xmask, tmp16_weight_next, tmp16_weight)
    tmp16_tmp, tmp17_tmp, tmp18_tmp = triton_helpers.welford(
        tmp16_mean, tmp16_m2, tmp16_weight, 1
    )
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp19 = tl.load(in_ptr0 + (x0 + (1500*r2) + (576000*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr2 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr3 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp41 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp19 + tmp20
        tmp22 = 0.5
        tmp23 = tmp21 * tmp22
        tmp24 = 0.7071067811865476
        tmp25 = tmp21 * tmp24
        tmp26 = libdevice.erf(tmp25)
        tmp27 = 1.0
        tmp28 = tmp26 + tmp27
        tmp29 = tmp23 * tmp28
        tmp31 = tmp29 + tmp30
        tmp33 = tmp31 + tmp32
        tmp34 = tmp33 - tmp16
        tmp35 = 384.0
        tmp36 = tmp17 / tmp35
        tmp37 = 1e-05
        tmp38 = tmp36 + tmp37
        tmp39 = libdevice.rsqrt(tmp38)
        tmp40 = tmp34 * tmp39
        tmp42 = tmp40 * tmp41
        tmp43 = tmp42 + tmp20
        tl.store(out_ptr2 + (r2 + (384*x3)), tmp43, rmask & xmask)
''')
