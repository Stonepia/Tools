

# Original file: ./cm3leon_generate__26_inference_66.6/cm3leon_generate__26_inference_66.6_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/ut/cutjyw4drovm3v2glsxxsv3ttfk5yoij7pjm2wofxkxzdzxmowch.py
# Source Nodes: [l__self___layers_0_self_attn_layer_norm, linear], Original ATen: [aten._to_copy, aten.native_layer_norm]
# l__self___layers_0_self_attn_layer_norm => add_2, add_3, mul_2, mul_3, rsqrt, sub_1, var_mean
# linear => convert_element_type_3
triton_red_fused__to_copy_native_layer_norm_1 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_1', '''
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
    size_hints=[1, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0 = tl.load(in_ptr0 + ((-1) + ks0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + ((-1) + ks0), None, eviction_policy='evict_last')
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp1 = tl.where(tmp0 < 0, tmp0 + 2048, tmp0)
        # tl.device_assert((0 <= tmp1) & (tmp1 < 2048), "index out of bounds: 0 <= tmp1 < 2048")
        tmp2 = tl.load(in_ptr1 + (r0 + (1536*tmp1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = 1.0
        tmp4 = tmp2 * tmp3
        tmp6 = tmp5.to(tl.int32)
        tmp7 = tl.full([1, 1], 2047, tl.int64)
        tmp8 = tmp0 != tmp7
        tmp9 = tmp8.to(tl.int32)
        tmp10 = tmp6 * tmp9
        tmp11 = tmp10.to(tl.int64)
        tmp12 = tmp11 + tmp7
        tmp13 = tl.where(tmp12 < 0, tmp12 + 4096, tmp12)
        # tl.device_assert((0 <= tmp13) & (tmp13 < 4096), "index out of bounds: 0 <= tmp13 < 4096")
        tmp14 = tl.load(in_ptr3 + (r0 + (1536*tmp13)), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp4 + tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_reduce(
            tmp16, tmp17_mean, tmp17_m2, tmp17_weight,
        )
        tmp17_mean = tl.where(rmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask, tmp17_weight_next, tmp17_weight)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp41 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp43 = tl.load(in_ptr5 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.where(tmp0 < 0, tmp0 + 2048, tmp0)
        # tl.device_assert((0 <= tmp20) & (tmp20 < 2048), "index out of bounds: 0 <= tmp20 < 2048")
        tmp21 = tl.load(in_ptr1 + (r0 + (1536*tmp20)), rmask, eviction_policy='evict_first', other=0.0)
        tmp22 = 1.0
        tmp23 = tmp21 * tmp22
        tmp24 = tmp5.to(tl.int32)
        tmp25 = tl.full([1, 1], 2047, tl.int64)
        tmp26 = tmp0 != tmp25
        tmp27 = tmp26.to(tl.int32)
        tmp28 = tmp24 * tmp27
        tmp29 = tmp28.to(tl.int64)
        tmp30 = tmp29 + tmp25
        tmp31 = tl.where(tmp30 < 0, tmp30 + 4096, tmp30)
        # tl.device_assert((0 <= tmp31) & (tmp31 < 4096), "index out of bounds: 0 <= tmp31 < 4096")
        tmp32 = tl.load(in_ptr3 + (r0 + (1536*tmp31)), rmask, eviction_policy='evict_first', other=0.0)
        tmp33 = tmp23 + tmp32
        tmp34 = tmp33 - tmp17
        tmp35 = 1536.0
        tmp36 = tmp18 / tmp35
        tmp37 = 1e-05
        tmp38 = tmp36 + tmp37
        tmp39 = libdevice.rsqrt(tmp38)
        tmp40 = tmp34 * tmp39
        tmp42 = tmp40 * tmp41
        tmp44 = tmp42 + tmp43
        tmp45 = tmp44.to(tl.float32)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp45, rmask)
''')
