

# Original file: ./poolformer_m36___60.0/poolformer_m36___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/3y/c3ye3tgm7a5i4xthqnds7cbqfkm7ng3xmiyu6oumxv3dty6sk6wq.py
# Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___1___token_mixer_pool, group_norm_15], Original ATen: [aten.avg_pool2d, aten.native_group_norm]
# getattr_getattr_l__mod___stages___1___blocks___1___token_mixer_pool => avg_pool2d_7
# group_norm_15 => convert_element_type_74, var_mean_15
triton_red_fused_avg_pool2d_native_group_norm_26 = async_compile.triton('triton_red_fused_avg_pool2d_native_group_norm_26', '''
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
    size_hints=[256, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_avg_pool2d_native_group_norm_26', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_avg_pool2d_native_group_norm_26(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 37632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    x6 = xindex % 4
    tmp88_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp88_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp88_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex // 28) % 28
        r1 = rindex % 28
        r5 = rindex
        tmp80 = tl.load(in_ptr1 + (r5 + (37632*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp81 = tl.load(in_ptr0 + (r5 + (37632*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp83 = tl.load(in_ptr2 + ((48*x6) + (r5 // 784)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp0 = (-1) + r2
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 28, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tmp2 & tmp4
        tmp6 = (-1) + r1
        tmp7 = tmp6 >= tmp1
        tmp8 = tmp6 < tmp3
        tmp9 = tmp7 & tmp8
        tmp10 = tmp5 & tmp9
        tmp11 = tl.load(in_ptr0 + ((-29) + r5 + (37632*x0)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.where(tmp10, tmp11, 0.0)
        tmp13 = r1
        tmp14 = tmp13 >= tmp1
        tmp15 = tmp13 < tmp3
        tmp16 = tmp14 & tmp15
        tmp17 = tmp5 & tmp16
        tmp18 = tl.load(in_ptr0 + ((-28) + r5 + (37632*x0)), rmask & tmp17 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tl.where(tmp17, tmp18, 0.0)
        tmp20 = tmp19 + tmp12
        tmp21 = 1 + r1
        tmp22 = tmp21 >= tmp1
        tmp23 = tmp21 < tmp3
        tmp24 = tmp22 & tmp23
        tmp25 = tmp5 & tmp24
        tmp26 = tl.load(in_ptr0 + ((-27) + r5 + (37632*x0)), rmask & tmp25 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp27 = tl.where(tmp25, tmp26, 0.0)
        tmp28 = tmp27 + tmp20
        tmp29 = r2
        tmp30 = tmp29 >= tmp1
        tmp31 = tmp29 < tmp3
        tmp32 = tmp30 & tmp31
        tmp33 = tmp32 & tmp9
        tmp34 = tl.load(in_ptr0 + ((-1) + r5 + (37632*x0)), rmask & tmp33 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tl.where(tmp33, tmp34, 0.0)
        tmp36 = tmp35 + tmp28
        tmp37 = tmp32 & tmp16
        tmp38 = tl.load(in_ptr0 + (r5 + (37632*x0)), rmask & tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp39 = tl.where(tmp37, tmp38, 0.0)
        tmp40 = tmp39 + tmp36
        tmp41 = tmp32 & tmp24
        tmp42 = tl.load(in_ptr0 + (1 + r5 + (37632*x0)), rmask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp43 = tl.where(tmp41, tmp42, 0.0)
        tmp44 = tmp43 + tmp40
        tmp45 = 1 + r2
        tmp46 = tmp45 >= tmp1
        tmp47 = tmp45 < tmp3
        tmp48 = tmp46 & tmp47
        tmp49 = tmp48 & tmp9
        tmp50 = tl.load(in_ptr0 + (27 + r5 + (37632*x0)), rmask & tmp49 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp51 = tl.where(tmp49, tmp50, 0.0)
        tmp52 = tmp51 + tmp44
        tmp53 = tmp48 & tmp16
        tmp54 = tl.load(in_ptr0 + (28 + r5 + (37632*x0)), rmask & tmp53 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp55 = tl.where(tmp53, tmp54, 0.0)
        tmp56 = tmp55 + tmp52
        tmp57 = tmp48 & tmp24
        tmp58 = tl.load(in_ptr0 + (29 + r5 + (37632*x0)), rmask & tmp57 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp59 = tl.where(tmp57, tmp58, 0.0)
        tmp60 = tmp59 + tmp56
        tmp61 = 1.0
        tmp62 = tl.where(tmp10, tmp61, 0.0)
        tmp63 = tl.where(tmp17, tmp61, 0.0)
        tmp64 = tmp63 + tmp62
        tmp65 = tl.where(tmp25, tmp61, 0.0)
        tmp66 = tmp65 + tmp64
        tmp67 = tl.where(tmp33, tmp61, 0.0)
        tmp68 = tmp67 + tmp66
        tmp69 = tl.where(tmp37, tmp61, 0.0)
        tmp70 = tmp69 + tmp68
        tmp71 = tl.where(tmp41, tmp61, 0.0)
        tmp72 = tmp71 + tmp70
        tmp73 = tl.where(tmp49, tmp61, 0.0)
        tmp74 = tmp73 + tmp72
        tmp75 = tl.where(tmp53, tmp61, 0.0)
        tmp76 = tmp75 + tmp74
        tmp77 = tl.where(tmp57, tmp61, 0.0)
        tmp78 = tmp77 + tmp76
        tmp79 = tmp60 / tmp78
        tmp82 = tmp79 - tmp81
        tmp84 = tmp82 * tmp83
        tmp85 = tmp80 + tmp84
        tmp86 = tmp85.to(tl.float32)
        tmp87 = tl.broadcast_to(tmp86, [XBLOCK, RBLOCK])
        tmp88_mean_next, tmp88_m2_next, tmp88_weight_next = triton_helpers.welford_reduce(
            tmp87, tmp88_mean, tmp88_m2, tmp88_weight,
        )
        tmp88_mean = tl.where(rmask & xmask, tmp88_mean_next, tmp88_mean)
        tmp88_m2 = tl.where(rmask & xmask, tmp88_m2_next, tmp88_m2)
        tmp88_weight = tl.where(rmask & xmask, tmp88_weight_next, tmp88_weight)
        tl.store(out_ptr0 + (r5 + (37632*x0)), tmp79, rmask & xmask)
    tmp88_tmp, tmp89_tmp, tmp90_tmp = triton_helpers.welford(
        tmp88_mean, tmp88_m2, tmp88_weight, 1
    )
    tmp88 = tmp88_tmp[:, None]
    tmp89 = tmp89_tmp[:, None]
    tmp90 = tmp90_tmp[:, None]
    tl.store(out_ptr1 + (x0), tmp88, xmask)
    tmp100_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp100_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp100_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        tmp91 = tl.load(in_ptr1 + (r5 + (37632*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp92 = tl.load(out_ptr0 + (r5 + (37632*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp93 = tl.load(in_ptr0 + (r5 + (37632*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp95 = tl.load(in_ptr2 + ((48*x6) + (r5 // 784)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp94 = tmp92 - tmp93
        tmp96 = tmp94 * tmp95
        tmp97 = tmp91 + tmp96
        tmp98 = tmp97.to(tl.float32)
        tmp99 = tl.broadcast_to(tmp98, [XBLOCK, RBLOCK])
        tmp100_mean_next, tmp100_m2_next, tmp100_weight_next = triton_helpers.welford_reduce(
            tmp99, tmp100_mean, tmp100_m2, tmp100_weight,
        )
        tmp100_mean = tl.where(rmask & xmask, tmp100_mean_next, tmp100_mean)
        tmp100_m2 = tl.where(rmask & xmask, tmp100_m2_next, tmp100_m2)
        tmp100_weight = tl.where(rmask & xmask, tmp100_weight_next, tmp100_weight)
    tmp100_tmp, tmp101_tmp, tmp102_tmp = triton_helpers.welford(
        tmp100_mean, tmp100_m2, tmp100_weight, 1
    )
    tmp100 = tmp100_tmp[:, None]
    tmp101 = tmp101_tmp[:, None]
    tmp102 = tmp102_tmp[:, None]
    tl.store(out_ptr2 + (x0), tmp101, xmask)
    tl.store(out_ptr3 + (x0), tmp102, xmask)
''')
