

# Original file: ./hf_GPT2_large___60.0/hf_GPT2_large___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/2n/c2nwhl5dggjymgwzwgkvvh4733gx35tyy2ojadweuwdh2leis4bx.py
# Source Nodes: [add_12, add_5, add_8, add_9, addmm_12, l__self___transformer_h_1_attn_resid_dropout, l__self___transformer_h_1_mlp_dropout, l__self___transformer_h_2_attn_resid_dropout, l__self___transformer_h_2_mlp_dropout, l__self___transformer_h_3_ln_1], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.native_layer_norm]
# add_12 => add_24
# add_5 => add_11
# add_8 => add_16
# add_9 => add_19
# addmm_12 => convert_element_type_43
# l__self___transformer_h_1_attn_resid_dropout => clone_7
# l__self___transformer_h_1_mlp_dropout => clone_8
# l__self___transformer_h_2_attn_resid_dropout => clone_11
# l__self___transformer_h_2_mlp_dropout => clone_12
# l__self___transformer_h_3_ln_1 => var_mean_6
triton_red_fused__to_copy_add_clone_native_layer_norm_12 = async_compile.triton('triton_red_fused__to_copy_add_clone_native_layer_norm_12', '''
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
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_native_layer_norm_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_native_layer_norm_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp14_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp14_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp14_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1280*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r1 + (1280*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1 + (1280*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r1 + (1280*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr4 + (r1 + (1280*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp5 + tmp7
        tmp9 = tmp1 + tmp8
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp9 + tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp14_mean_next, tmp14_m2_next, tmp14_weight_next = triton_helpers.welford_reduce(
            tmp13, tmp14_mean, tmp14_m2, tmp14_weight,
        )
        tmp14_mean = tl.where(rmask & xmask, tmp14_mean_next, tmp14_mean)
        tmp14_m2 = tl.where(rmask & xmask, tmp14_m2_next, tmp14_m2)
        tmp14_weight = tl.where(rmask & xmask, tmp14_weight_next, tmp14_weight)
        tl.store(out_ptr0 + (r1 + (1280*x0)), tmp12, rmask & xmask)
    tmp14_tmp, tmp15_tmp, tmp16_tmp = triton_helpers.welford(
        tmp14_mean, tmp14_m2, tmp14_weight, 1
    )
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp19_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp17 = tl.load(out_ptr0 + (r1 + (1280*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        r1 = rindex
        tmp22 = tl.load(out_ptr0 + (r1 + (1280*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp30 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tmp22 - tmp14
        tmp24 = 1280.0
        tmp25 = tmp20 / tmp24
        tmp26 = 1e-05
        tmp27 = tmp25 + tmp26
        tmp28 = libdevice.rsqrt(tmp27)
        tmp29 = tmp23 * tmp28
        tmp31 = tmp29 * tmp30
        tmp33 = tmp31 + tmp32
        tmp34 = tmp33.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (1280*x0)), tmp34, rmask & xmask)
''')
