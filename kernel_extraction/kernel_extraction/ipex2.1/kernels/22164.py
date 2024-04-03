

# Original file: ./gmlp_s16_224___60.0/gmlp_s16_224___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/th/cthsdbeyxcxcno6rdypxvxwp45eeumt453iu4ykwymxb22hwr7pi.py
# Source Nodes: [getattr_l__self___blocks___0___mlp_channels_gate_norm], Original ATen: [aten.native_layer_norm]
# getattr_l__self___blocks___0___mlp_channels_gate_norm => add_3, add_4, clone_2, convert_element_type_10, convert_element_type_9, mul_5, mul_6, rsqrt_1, sub_1, var_mean_1
triton_red_fused_native_layer_norm_2 = async_compile.triton('triton_red_fused_native_layer_norm_2', '''
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
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp13_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (768 + r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = 0.5
        tmp3 = tmp1 * tmp2
        tmp4 = 0.7071067811865476
        tmp5 = tmp1 * tmp4
        tmp6 = libdevice.erf(tmp5)
        tmp7 = 1.0
        tmp8 = tmp6 + tmp7
        tmp9 = tmp3 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_reduce(
            tmp12, tmp13_mean, tmp13_m2, tmp13_weight,
        )
        tmp13_mean = tl.where(rmask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(rmask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(rmask & xmask, tmp13_weight_next, tmp13_weight)
    tmp13_tmp, tmp14_tmp, tmp15_tmp = triton_helpers.welford(
        tmp13_mean, tmp13_m2, tmp13_weight, 1
    )
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(in_ptr0 + (768 + r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = 0.5
        tmp19 = tmp17 * tmp18
        tmp20 = 0.7071067811865476
        tmp21 = tmp17 * tmp20
        tmp22 = libdevice.erf(tmp21)
        tmp23 = 1.0
        tmp24 = tmp22 + tmp23
        tmp25 = tmp19 * tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27 - tmp13
        tmp29 = 768.0
        tmp30 = tmp14 / tmp29
        tmp31 = 1e-05
        tmp32 = tmp30 + tmp31
        tmp33 = libdevice.rsqrt(tmp32)
        tmp34 = tmp28 * tmp33
        tmp36 = tmp34 * tmp35
        tmp38 = tmp36 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0)), tmp39, rmask & xmask)
''')
