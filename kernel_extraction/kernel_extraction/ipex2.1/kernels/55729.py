

# Original file: ./cm3leon_generate__30_inference_70.10/cm3leon_generate__30_inference_70.10_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/ib/cibpvfwa6kxrjclnjeendbcfewj7zl2yv6ryrkez7x4g7pi4cvrn.py
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
    meta={'signature': {0: '*i64', 1: '*fp16', 2: '*i64', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0 = tl.load(in_ptr0 + ((-1) + ks0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + ((-1) + ks0), None, eviction_policy='evict_last')
    tmp18_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp1 = tl.where(tmp0 < 0, tmp0 + 2048, tmp0)
        # tl.device_assert((0 <= tmp1) & (tmp1 < 2048), "index out of bounds: 0 <= tmp1 < 2048")
        tmp2 = tl.load(in_ptr1 + (r0 + (1536*tmp1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp14 = tl.load(in_ptr3 + (r0 + (1536*tmp13)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp4 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp18_mean_next, tmp18_m2_next, tmp18_weight_next = triton_helpers.welford_reduce(
            tmp17, tmp18_mean, tmp18_m2, tmp18_weight,
        )
        tmp18_mean = tl.where(rmask, tmp18_mean_next, tmp18_mean)
        tmp18_m2 = tl.where(rmask, tmp18_m2_next, tmp18_m2)
        tmp18_weight = tl.where(rmask, tmp18_weight_next, tmp18_weight)
    tmp18_tmp, tmp19_tmp, tmp20_tmp = triton_helpers.welford(
        tmp18_mean, tmp18_m2, tmp18_weight, 1
    )
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp43 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp46 = tl.load(in_ptr5 + (r0), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.where(tmp0 < 0, tmp0 + 2048, tmp0)
        # tl.device_assert((0 <= tmp21) & (tmp21 < 2048), "index out of bounds: 0 <= tmp21 < 2048")
        tmp22 = tl.load(in_ptr1 + (r0 + (1536*tmp21)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = 1.0
        tmp24 = tmp22 * tmp23
        tmp25 = tmp5.to(tl.int32)
        tmp26 = tl.full([1, 1], 2047, tl.int64)
        tmp27 = tmp0 != tmp26
        tmp28 = tmp27.to(tl.int32)
        tmp29 = tmp25 * tmp28
        tmp30 = tmp29.to(tl.int64)
        tmp31 = tmp30 + tmp26
        tmp32 = tl.where(tmp31 < 0, tmp31 + 4096, tmp31)
        # tl.device_assert((0 <= tmp32) & (tmp32 < 4096), "index out of bounds: 0 <= tmp32 < 4096")
        tmp33 = tl.load(in_ptr3 + (r0 + (1536*tmp32)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp34 = tmp24 + tmp33
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp35 - tmp18
        tmp37 = 1536.0
        tmp38 = tmp19 / tmp37
        tmp39 = 1e-05
        tmp40 = tmp38 + tmp39
        tmp41 = libdevice.rsqrt(tmp40)
        tmp42 = tmp36 * tmp41
        tmp44 = tmp43.to(tl.float32)
        tmp45 = tmp42 * tmp44
        tmp47 = tmp46.to(tl.float32)
        tmp48 = tmp45 + tmp47
        tmp49 = tmp48.to(tl.float32)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp49, rmask)
''')
