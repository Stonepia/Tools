

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/df/cdft6touaf7y6tmprxzaoi5qxqyb6thft6sobgeapzpaihjoutfz.py
# Source Nodes: [add_46, add_47, add_48, l__self___layer_norm], Original ATen: [aten.add, aten.native_layer_norm]
# add_46 => add_162
# add_47 => add_166
# add_48 => add_169
# l__self___layer_norm => add_170, add_171, mul_122, mul_123, rsqrt_48, sub_73, var_mean_48
triton_red_fused_add_native_layer_norm_86 = async_compile.triton('triton_red_fused_add_native_layer_norm_86', '''
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
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_86', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_86(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp11_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp6 + tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp11_mean_next, tmp11_m2_next, tmp11_weight_next = triton_helpers.welford_reduce(
            tmp10, tmp11_mean, tmp11_m2, tmp11_weight,
        )
        tmp11_mean = tl.where(rmask, tmp11_mean_next, tmp11_mean)
        tmp11_m2 = tl.where(rmask, tmp11_m2_next, tmp11_m2)
        tmp11_weight = tl.where(rmask, tmp11_weight_next, tmp11_weight)
    tmp11_tmp, tmp12_tmp, tmp13_tmp = triton_helpers.welford(
        tmp11_mean, tmp11_m2, tmp11_weight, 1
    )
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp14 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr1 + (r0), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp33 = tl.load(in_ptr5 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp14 + tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 + tmp19
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 + tmp22
        tmp24 = tmp23 - tmp11
        tmp25 = 1536.0
        tmp26 = tmp12 / tmp25
        tmp27 = 1e-05
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp30 * tmp31
        tmp34 = tmp32 + tmp33
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp34, rmask)
''')
