

# Original file: ./hf_Whisper___60.0/hf_Whisper___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/bj/cbje3owklbqyy7fjwtdptwp5qohcvccpyjyvqzjuugdruduinmyi.py
# Source Nodes: [add, l__mod___encoder_layers_0_self_attn_layer_norm], Original ATen: [aten.add, aten.native_layer_norm]
# add => add_2
# l__mod___encoder_layers_0_self_attn_layer_norm => add_3, add_4, clone_1, convert_element_type_4, convert_element_type_5, mul_6, mul_7, rsqrt, sub, var_mean
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
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
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1500*r2) + (576000*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr2 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 0.5
        tmp5 = tmp3 * tmp4
        tmp6 = 0.7071067811865476
        tmp7 = tmp3 * tmp6
        tmp8 = libdevice.erf(tmp7)
        tmp9 = 1.0
        tmp10 = tmp8 + tmp9
        tmp11 = tmp5 * tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp14 = tmp12 + tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_reduce(
            tmp16, tmp17_mean, tmp17_m2, tmp17_weight,
        )
        tmp17_mean = tl.where(rmask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask & xmask, tmp17_weight_next, tmp17_weight)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp20 = tl.load(in_ptr0 + (x0 + (1500*r2) + (576000*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp43 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tmp20 + tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = 0.5
        tmp25 = tmp23 * tmp24
        tmp26 = 0.7071067811865476
        tmp27 = tmp23 * tmp26
        tmp28 = libdevice.erf(tmp27)
        tmp29 = 1.0
        tmp30 = tmp28 + tmp29
        tmp31 = tmp25 * tmp30
        tmp32 = tmp31.to(tl.float32)
        tmp34 = tmp32 + tmp33
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp35 - tmp17
        tmp37 = 384.0
        tmp38 = tmp18 / tmp37
        tmp39 = 1e-05
        tmp40 = tmp38 + tmp39
        tmp41 = libdevice.rsqrt(tmp40)
        tmp42 = tmp36 * tmp41
        tmp44 = tmp43.to(tl.float32)
        tmp45 = tmp42 * tmp44
        tmp46 = tmp21.to(tl.float32)
        tmp47 = tmp45 + tmp46
        tmp48 = tmp47.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (384*x3)), tmp48, rmask & xmask)
''')
