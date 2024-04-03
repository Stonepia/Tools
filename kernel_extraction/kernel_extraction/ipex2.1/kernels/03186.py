

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/6a/c6ahioyrvgryhokuz2mxrrm3l43dngfcn44cen2gwviizx3r3pf7.py
# Source Nodes: [l__self___layers_0_self_attn_layer_norm], Original ATen: [aten.native_layer_norm]
# l__self___layers_0_self_attn_layer_norm => add_2, add_3, convert_element_type_4, convert_element_type_5, mul_2, mul_3, rsqrt, sub_1, var_mean
triton_red_fused_native_layer_norm_1 = async_compile.triton('triton_red_fused_native_layer_norm_1', '''
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
    meta={'signature': {0: '*i64', 1: '*bf16', 2: '*i64', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp20_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp2 = tl.where(tmp1 < 0, tmp1 + 2048, tmp1)
        # tl.device_assert((0 <= tmp2) & (tmp2 < 2048), "index out of bounds: 0 <= tmp2 < 2048")
        tmp3 = tl.load(in_ptr1 + (r0 + (1536*tmp2)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp16 = tl.load(in_ptr3 + (r0 + (1536*tmp15)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tmp5 + tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp20_mean_next, tmp20_m2_next, tmp20_weight_next = triton_helpers.welford_reduce(
            tmp19, tmp20_mean, tmp20_m2, tmp20_weight,
        )
        tmp20_mean = tl.where(rmask, tmp20_mean_next, tmp20_mean)
        tmp20_m2 = tl.where(rmask, tmp20_m2_next, tmp20_m2)
        tmp20_weight = tl.where(rmask, tmp20_weight_next, tmp20_weight)
    tmp20_tmp, tmp21_tmp, tmp22_tmp = triton_helpers.welford(
        tmp20_mean, tmp20_m2, tmp20_weight, 1
    )
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tl.load(in_ptr0 + (63))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp29 = tl.load(in_ptr2 + (63))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp49 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp52 = tl.load(in_ptr5 + (r0), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tl.where(tmp24 < 0, tmp24 + 2048, tmp24)
        # tl.device_assert((0 <= tmp25) & (tmp25 < 2048), "index out of bounds: 0 <= tmp25 < 2048")
        tmp26 = tl.load(in_ptr1 + (r0 + (1536*tmp25)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = 1.0
        tmp28 = tmp26 * tmp27
        tmp31 = tmp30.to(tl.int32)
        tmp32 = tl.full([1, 1], 2047, tl.int64)
        tmp33 = tmp24 != tmp32
        tmp34 = tmp33.to(tl.int32)
        tmp35 = tmp31 * tmp34
        tmp36 = tmp35.to(tl.int64)
        tmp37 = tmp36 + tmp32
        tmp38 = tl.where(tmp37 < 0, tmp37 + 4096, tmp37)
        # tl.device_assert((0 <= tmp38) & (tmp38 < 4096), "index out of bounds: 0 <= tmp38 < 4096")
        tmp39 = tl.load(in_ptr3 + (r0 + (1536*tmp38)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp40 = tmp28 + tmp39
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tmp41 - tmp20
        tmp43 = 1536.0
        tmp44 = tmp21 / tmp43
        tmp45 = 1e-05
        tmp46 = tmp44 + tmp45
        tmp47 = libdevice.rsqrt(tmp46)
        tmp48 = tmp42 * tmp47
        tmp50 = tmp49.to(tl.float32)
        tmp51 = tmp48 * tmp50
        tmp53 = tmp52.to(tl.float32)
        tmp54 = tmp51 + tmp53
        tmp55 = tmp54.to(tl.float32)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp55, rmask)
''')
