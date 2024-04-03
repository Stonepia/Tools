

# Original file: ./cm3leon_generate__30_inference_70.10/cm3leon_generate__30_inference_70.10_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/mn/cmny4mkpqazngrk2pfr7opf7tdvsspm3xmyqtnnfuidzxsmjzuak.py
# Source Nodes: [add_1, l__self___layers_0_final_layer_norm, linear_4], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm]
# add_1 => add_5
# l__self___layers_0_final_layer_norm => add_6, add_7, mul_5, mul_6, rsqrt_1, sub_3, var_mean_1
# linear_4 => convert_element_type_19
triton_red_fused__to_copy_add_native_layer_norm_7 = async_compile.triton('triton_red_fused__to_copy_add_native_layer_norm_7', '''
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
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_native_layer_norm_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 11), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_native_layer_norm_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0 = tl.load(in_ptr0 + ((-1) + ks0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + ((-1) + ks0), None, eviction_policy='evict_last')
    tmp20_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp16 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.where(tmp0 < 0, tmp0 + 2048, tmp0)
        # tl.device_assert((0 <= tmp1) & (tmp1 < 2048), "index out of bounds: 0 <= tmp1 < 2048")
        tmp2 = tl.load(in_ptr1 + (r0 + (1536*tmp1)), rmask, eviction_policy='evict_first', other=0.0)
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
        tmp14 = tl.load(in_ptr3 + (r0 + (1536*tmp13)), rmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tmp4 + tmp14
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp15 + tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp20_mean_next, tmp20_m2_next, tmp20_weight_next = triton_helpers.welford_reduce(
            tmp19, tmp20_mean, tmp20_m2, tmp20_weight,
        )
        tmp20_mean = tl.where(rmask, tmp20_mean_next, tmp20_mean)
        tmp20_m2 = tl.where(rmask, tmp20_m2_next, tmp20_m2)
        tmp20_weight = tl.where(rmask, tmp20_weight_next, tmp20_weight)
        tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp18, rmask)
    tmp20_tmp, tmp21_tmp, tmp22_tmp = triton_helpers.welford(
        tmp20_mean, tmp20_m2, tmp20_weight, 1
    )
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tmp25_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp25_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp25_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp23 = tl.load(out_ptr0 + (r0), rmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp25_mean_next, tmp25_m2_next, tmp25_weight_next = triton_helpers.welford_reduce(
            tmp24, tmp25_mean, tmp25_m2, tmp25_weight,
        )
        tmp25_mean = tl.where(rmask, tmp25_mean_next, tmp25_mean)
        tmp25_m2 = tl.where(rmask, tmp25_m2_next, tmp25_m2)
        tmp25_weight = tl.where(rmask, tmp25_weight_next, tmp25_weight)
    tmp25_tmp, tmp26_tmp, tmp27_tmp = triton_helpers.welford(
        tmp25_mean, tmp25_m2, tmp25_weight, 1
    )
    tmp25 = tmp25_tmp[:, None]
    tmp26 = tmp26_tmp[:, None]
    tmp27 = tmp27_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp28 = tl.load(out_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp36 = tl.load(in_ptr5 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp38 = tl.load(in_ptr6 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp29 = tmp28 - tmp20
        tmp30 = 1536.0
        tmp31 = tmp26 / tmp30
        tmp32 = 1e-05
        tmp33 = tmp31 + tmp32
        tmp34 = libdevice.rsqrt(tmp33)
        tmp35 = tmp29 * tmp34
        tmp37 = tmp35 * tmp36
        tmp39 = tmp37 + tmp38
        tmp40 = tmp39.to(tl.float32)
        tl.store(out_ptr3 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp40, rmask)
''')
