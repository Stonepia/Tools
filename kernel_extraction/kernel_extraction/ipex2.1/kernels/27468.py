

# Original file: ./cm3leon_generate__28_inference_68.8/cm3leon_generate__28_inference_68.8_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/n7/cn7z6x7cvpmcmh3333kjn6dexdcz73e5b6exrtlhqruxlqlr7pyl.py
# Source Nodes: [add_1, l__self___layers_0_final_layer_norm], Original ATen: [aten.add, aten.native_layer_norm]
# add_1 => add_5
# l__self___layers_0_final_layer_norm => add_6, add_7, mul_5, mul_6, rsqrt_1, sub_3, var_mean_1
triton_red_fused_add_native_layer_norm_7 = async_compile.triton('triton_red_fused_add_native_layer_norm_7', '''
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
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 10), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0 = tl.load(in_ptr0 + ((-1) + ks0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + ((-1) + ks0), None, eviction_policy='evict_last')
    tmp19_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp16 = tl.load(in_out_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
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
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp19_mean_next, tmp19_m2_next, tmp19_weight_next = triton_helpers.welford_reduce(
            tmp18, tmp19_mean, tmp19_m2, tmp19_weight,
        )
        tmp19_mean = tl.where(rmask, tmp19_mean_next, tmp19_mean)
        tmp19_m2 = tl.where(rmask, tmp19_m2_next, tmp19_m2)
        tmp19_weight = tl.where(rmask, tmp19_weight_next, tmp19_weight)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp17, rmask)
    tmp19_tmp, tmp20_tmp, tmp21_tmp = triton_helpers.welford(
        tmp19_mean, tmp19_m2, tmp19_weight, 1
    )
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tmp24_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp22 = tl.load(in_out_ptr0 + (r0), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp24_mean_next, tmp24_m2_next, tmp24_weight_next = triton_helpers.welford_reduce(
            tmp23, tmp24_mean, tmp24_m2, tmp24_weight,
        )
        tmp24_mean = tl.where(rmask, tmp24_mean_next, tmp24_mean)
        tmp24_m2 = tl.where(rmask, tmp24_m2_next, tmp24_m2)
        tmp24_weight = tl.where(rmask, tmp24_weight_next, tmp24_weight)
    tmp24_tmp, tmp25_tmp, tmp26_tmp = triton_helpers.welford(
        tmp24_mean, tmp24_m2, tmp24_weight, 1
    )
    tmp24 = tmp24_tmp[:, None]
    tmp25 = tmp25_tmp[:, None]
    tmp26 = tmp26_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp27 = tl.load(in_out_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp35 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp37 = tl.load(in_ptr5 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tmp27 - tmp19
        tmp29 = 1536.0
        tmp30 = tmp25 / tmp29
        tmp31 = 1e-05
        tmp32 = tmp30 + tmp31
        tmp33 = libdevice.rsqrt(tmp32)
        tmp34 = tmp28 * tmp33
        tmp36 = tmp34 * tmp35
        tmp38 = tmp36 + tmp37
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp38, rmask)
''')
