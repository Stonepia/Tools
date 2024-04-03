

# Original file: ./hf_Whisper___60.0/hf_Whisper___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ns/cnskiut2zdtq4fjqazk3xluwkuvhpiomszjsad7x7nrctddffag6.py
# Source Nodes: [add, add_1, dropout_2, l__mod___encoder_layers_0_final_layer_norm], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add => add_2
# add_1 => add_5
# dropout_2 => clone_7
# l__mod___encoder_layers_0_final_layer_norm => add_6, add_7, clone_8, convert_element_type_8, convert_element_type_9, mul_10, mul_9, rsqrt_1, sub_2, var_mean_1
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_native_layer_norm_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
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
    tmp19_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1500*r2) + (576000*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr2 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr3 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp16 = tmp14 + tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp19_mean_next, tmp19_m2_next, tmp19_weight_next = triton_helpers.welford_reduce(
            tmp18, tmp19_mean, tmp19_m2, tmp19_weight,
        )
        tmp19_mean = tl.where(rmask & xmask, tmp19_mean_next, tmp19_mean)
        tmp19_m2 = tl.where(rmask & xmask, tmp19_m2_next, tmp19_m2)
        tmp19_weight = tl.where(rmask & xmask, tmp19_weight_next, tmp19_weight)
    tmp19_tmp, tmp20_tmp, tmp21_tmp = triton_helpers.welford(
        tmp19_mean, tmp19_m2, tmp19_weight, 1
    )
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp22 = tl.load(in_ptr0 + (x0 + (1500*r2) + (576000*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr2 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp37 = tl.load(in_ptr3 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp47 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp24 = tmp22 + tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp26 = 0.5
        tmp27 = tmp25 * tmp26
        tmp28 = 0.7071067811865476
        tmp29 = tmp25 * tmp28
        tmp30 = libdevice.erf(tmp29)
        tmp31 = 1.0
        tmp32 = tmp30 + tmp31
        tmp33 = tmp27 * tmp32
        tmp34 = tmp33.to(tl.float32)
        tmp36 = tmp34 + tmp35
        tmp38 = tmp36 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp39 - tmp19
        tmp41 = 384.0
        tmp42 = tmp20 / tmp41
        tmp43 = 1e-05
        tmp44 = tmp42 + tmp43
        tmp45 = libdevice.rsqrt(tmp44)
        tmp46 = tmp40 * tmp45
        tmp48 = tmp47.to(tl.float32)
        tmp49 = tmp46 * tmp48
        tmp50 = tmp23.to(tl.float32)
        tmp51 = tmp49 + tmp50
        tmp52 = tmp51.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (384*x3)), tmp52, rmask & xmask)
''')
