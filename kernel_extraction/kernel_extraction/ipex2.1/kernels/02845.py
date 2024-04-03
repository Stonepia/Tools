

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/2z/c2zn3vncio3rzd24oitctsxgc5roypguxyenjljumxmdypg3peuq.py
# Source Nodes: [add_1, l__self___layers_0_final_layer_norm], Original ATen: [aten.add, aten.native_layer_norm]
# add_1 => add_5
# l__self___layers_0_final_layer_norm => add_6, add_7, mul_5, mul_6, rsqrt_1, sub_3, var_mean_1
triton_red_fused_add_native_layer_norm_6 = async_compile.triton('triton_red_fused_add_native_layer_norm_6', '''
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
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp21_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp18 = tl.load(in_out_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
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
        tmp19 = tmp17 + tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp21_mean_next, tmp21_m2_next, tmp21_weight_next = triton_helpers.welford_reduce(
            tmp20, tmp21_mean, tmp21_m2, tmp21_weight,
        )
        tmp21_mean = tl.where(rmask, tmp21_mean_next, tmp21_mean)
        tmp21_m2 = tl.where(rmask, tmp21_m2_next, tmp21_m2)
        tmp21_weight = tl.where(rmask, tmp21_weight_next, tmp21_weight)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp19, rmask)
    tmp21_tmp, tmp22_tmp, tmp23_tmp = triton_helpers.welford(
        tmp21_mean, tmp21_m2, tmp21_weight, 1
    )
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tmp26_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp26_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp26_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp24 = tl.load(in_out_ptr0 + (r0), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp26_mean_next, tmp26_m2_next, tmp26_weight_next = triton_helpers.welford_reduce(
            tmp25, tmp26_mean, tmp26_m2, tmp26_weight,
        )
        tmp26_mean = tl.where(rmask, tmp26_mean_next, tmp26_mean)
        tmp26_m2 = tl.where(rmask, tmp26_m2_next, tmp26_m2)
        tmp26_weight = tl.where(rmask, tmp26_weight_next, tmp26_weight)
    tmp26_tmp, tmp27_tmp, tmp28_tmp = triton_helpers.welford(
        tmp26_mean, tmp26_m2, tmp26_weight, 1
    )
    tmp26 = tmp26_tmp[:, None]
    tmp27 = tmp27_tmp[:, None]
    tmp28 = tmp28_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp37 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp39 = tl.load(in_ptr5 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp30 = tmp29 - tmp21
        tmp31 = 1536.0
        tmp32 = tmp27 / tmp31
        tmp33 = 1e-05
        tmp34 = tmp32 + tmp33
        tmp35 = libdevice.rsqrt(tmp34)
        tmp36 = tmp30 * tmp35
        tmp38 = tmp36 * tmp37
        tmp40 = tmp38 + tmp39
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp40, rmask)
''')
