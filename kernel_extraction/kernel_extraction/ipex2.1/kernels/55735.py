

# Original file: ./cm3leon_generate__30_inference_70.10/cm3leon_generate__30_inference_70.10_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/dj/cdjrwbyzhv5rggss2tscjtbcns4dfroxqg7fxhsshpuh5u7m6lkg.py
# Source Nodes: [add_1, l__self___layers_0_final_layer_norm], Original ATen: [aten.add, aten.native_layer_norm]
# add_1 => add_5
# l__self___layers_0_final_layer_norm => add_6, add_7, convert_element_type_10, convert_element_type_11, mul_5, mul_6, rsqrt_1, sub_3, var_mean_1
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
    meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*i64', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 10), equal_to_1=())]}
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
    tmp20_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp16 = tl.load(in_out_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.where(tmp0 < 0, tmp0 + 2048, tmp0)
        # tl.device_assert((0 <= tmp1) & (tmp1 < 2048), "index out of bounds: 0 <= tmp1 < 2048")
        tmp2 = tl.load(in_ptr1 + (r0 + (1536*tmp1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
        tmp14 = tl.load(in_ptr3 + (r0 + (1536*tmp13)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tmp4 + tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp20_mean_next, tmp20_m2_next, tmp20_weight_next = triton_helpers.welford_reduce(
            tmp19, tmp20_mean, tmp20_m2, tmp20_weight,
        )
        tmp20_mean = tl.where(rmask, tmp20_mean_next, tmp20_mean)
        tmp20_m2 = tl.where(rmask, tmp20_m2_next, tmp20_m2)
        tmp20_weight = tl.where(rmask, tmp20_weight_next, tmp20_weight)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp17, rmask)
    tmp20_tmp, tmp21_tmp, tmp22_tmp = triton_helpers.welford(
        tmp20_mean, tmp20_m2, tmp20_weight, 1
    )
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tmp26_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp26_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp26_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp23 = tl.load(in_out_ptr0 + (r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp24 = tmp23.to(tl.float32)
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
        tmp29 = tl.load(in_out_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp38 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp41 = tl.load(in_ptr5 + (r0), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp20
        tmp32 = 1536.0
        tmp33 = tmp27 / tmp32
        tmp34 = 1e-05
        tmp35 = tmp33 + tmp34
        tmp36 = libdevice.rsqrt(tmp35)
        tmp37 = tmp31 * tmp36
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp37 * tmp39
        tmp42 = tmp41.to(tl.float32)
        tmp43 = tmp40 + tmp42
        tmp44 = tmp43.to(tl.float32)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp44, rmask)
''')
