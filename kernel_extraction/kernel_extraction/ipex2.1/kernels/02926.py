

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/aj/cajglmkcvcjh6ngptq45b4tbjrj3vcsiwyz6qwa7hblah5ca6jcj.py
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
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp19_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp2 = tl.where(tmp1 < 0, tmp1 + 2048, tmp1)
        # tl.device_assert((0 <= tmp2) & (tmp2 < 2048), "index out of bounds: 0 <= tmp2 < 2048")
        tmp3 = tl.load(in_ptr1 + (r0 + (1536*tmp2)), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp16 = tl.load(in_ptr3 + (r0 + (1536*tmp15)), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp5 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp19_mean_next, tmp19_m2_next, tmp19_weight_next = triton_helpers.welford_reduce(
            tmp18, tmp19_mean, tmp19_m2, tmp19_weight,
        )
        tmp19_mean = tl.where(rmask, tmp19_mean_next, tmp19_mean)
        tmp19_m2 = tl.where(rmask, tmp19_m2_next, tmp19_m2)
        tmp19_weight = tl.where(rmask, tmp19_weight_next, tmp19_weight)
    tmp19_tmp, tmp20_tmp, tmp21_tmp = triton_helpers.welford(
        tmp19_mean, tmp19_m2, tmp19_weight, 1
    )
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tl.load(in_ptr0 + (63))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp28 = tl.load(in_ptr2 + (63))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp47 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp49 = tl.load(in_ptr5 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.where(tmp23 < 0, tmp23 + 2048, tmp23)
        # tl.device_assert((0 <= tmp24) & (tmp24 < 2048), "index out of bounds: 0 <= tmp24 < 2048")
        tmp25 = tl.load(in_ptr1 + (r0 + (1536*tmp24)), rmask, eviction_policy='evict_first', other=0.0)
        tmp26 = 1.0
        tmp27 = tmp25 * tmp26
        tmp30 = tmp29.to(tl.int32)
        tmp31 = tl.full([1, 1], 2047, tl.int64)
        tmp32 = tmp23 != tmp31
        tmp33 = tmp32.to(tl.int32)
        tmp34 = tmp30 * tmp33
        tmp35 = tmp34.to(tl.int64)
        tmp36 = tmp35 + tmp31
        tmp37 = tl.where(tmp36 < 0, tmp36 + 4096, tmp36)
        # tl.device_assert((0 <= tmp37) & (tmp37 < 4096), "index out of bounds: 0 <= tmp37 < 4096")
        tmp38 = tl.load(in_ptr3 + (r0 + (1536*tmp37)), rmask, eviction_policy='evict_first', other=0.0)
        tmp39 = tmp27 + tmp38
        tmp40 = tmp39 - tmp19
        tmp41 = 1536.0
        tmp42 = tmp20 / tmp41
        tmp43 = 1e-05
        tmp44 = tmp42 + tmp43
        tmp45 = libdevice.rsqrt(tmp44)
        tmp46 = tmp40 * tmp45
        tmp48 = tmp46 * tmp47
        tmp50 = tmp48 + tmp49
        tmp51 = tmp50.to(tl.float32)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp51, rmask)
''')
