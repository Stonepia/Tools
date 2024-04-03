

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/35/c357wtqjq2dkfcspbpj7vrpp42x4pfqxvkbgs4x5min7zwnosc3n.py
# Source Nodes: [add_1, l__self___layers_0_final_layer_norm, linear_4], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm]
# add_1 => add_5
# l__self___layers_0_final_layer_norm => add_6, add_7, mul_5, mul_6, rsqrt_1, sub_3, var_mean_1
# linear_4 => convert_element_type_19
triton_red_fused__to_copy_add_native_layer_norm_6 = async_compile.triton('triton_red_fused__to_copy_add_native_layer_norm_6', '''
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
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*bf16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_native_layer_norm_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_native_layer_norm_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0 = tl.load(in_ptr0 + (63))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr2 + (63))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp22_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp18 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.where(tmp1 < 0, tmp1 + 2048, tmp1)
        # tl.device_assert((0 <= tmp2) & (tmp2 < 2048), "index out of bounds: 0 <= tmp2 < 2048")
        tmp3 = tl.load(in_ptr1 + (r0 + (1536*tmp2)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 1.0
        tmp5 = tmp3 * tmp4
        tmp8 = tmp7.to(tl.int32)
        tmp9 = tl.full([1, 1], 2047, tl.int64)
        tmp10 = tmp1 != tmp9
        tmp11 = tmp10.to(tl.int32)
        tmp12 = tmp8 * tmp11
        tmp13 = tmp12.to(tl.int64)
        tmp14 = tmp13 + tmp9
        tmp15 = tl.where(tmp14 < 0, tmp14 + 4096, tmp14)
        # tl.device_assert((0 <= tmp15) & (tmp15 < 4096), "index out of bounds: 0 <= tmp15 < 4096")
        tmp16 = tl.load(in_ptr3 + (r0 + (1536*tmp15)), rmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tmp5 + tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 + tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp22_mean_next, tmp22_m2_next, tmp22_weight_next = triton_helpers.welford_reduce(
            tmp21, tmp22_mean, tmp22_m2, tmp22_weight,
        )
        tmp22_mean = tl.where(rmask, tmp22_mean_next, tmp22_mean)
        tmp22_m2 = tl.where(rmask, tmp22_m2_next, tmp22_m2)
        tmp22_weight = tl.where(rmask, tmp22_weight_next, tmp22_weight)
        tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp20, rmask)
    tmp22_tmp, tmp23_tmp, tmp24_tmp = triton_helpers.welford(
        tmp22_mean, tmp22_m2, tmp22_weight, 1
    )
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tmp24 = tmp24_tmp[:, None]
    tmp27_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp27_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp27_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp25 = tl.load(out_ptr0 + (r0), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp27_mean_next, tmp27_m2_next, tmp27_weight_next = triton_helpers.welford_reduce(
            tmp26, tmp27_mean, tmp27_m2, tmp27_weight,
        )
        tmp27_mean = tl.where(rmask, tmp27_mean_next, tmp27_mean)
        tmp27_m2 = tl.where(rmask, tmp27_m2_next, tmp27_m2)
        tmp27_weight = tl.where(rmask, tmp27_weight_next, tmp27_weight)
    tmp27_tmp, tmp28_tmp, tmp29_tmp = triton_helpers.welford(
        tmp27_mean, tmp27_m2, tmp27_weight, 1
    )
    tmp27 = tmp27_tmp[:, None]
    tmp28 = tmp28_tmp[:, None]
    tmp29 = tmp29_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp30 = tl.load(out_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp38 = tl.load(in_ptr5 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp40 = tl.load(in_ptr6 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp31 = tmp30 - tmp22
        tmp32 = 1536.0
        tmp33 = tmp28 / tmp32
        tmp34 = 1e-05
        tmp35 = tmp33 + tmp34
        tmp36 = libdevice.rsqrt(tmp35)
        tmp37 = tmp31 * tmp36
        tmp39 = tmp37 * tmp38
        tmp41 = tmp39 + tmp40
        tmp42 = tmp41.to(tl.float32)
        tl.store(out_ptr3 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp42, rmask)
''')
