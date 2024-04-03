

# Original file: ./twins_pcpvt_base___60.0/twins_pcpvt_base___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/dy/cdygspho2swzh6vj7evablop7s54tatff3ziugz6x2yw3rttol4c.py
# Source Nodes: [l__self___blocks_0_0_norm1, l__self___patch_embeds_0_norm], Original ATen: [aten.native_layer_norm]
# l__self___blocks_0_0_norm1 => add_2, add_3, convert_element_type_5, convert_element_type_6, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# l__self___patch_embeds_0_norm => add, add_1, clone, convert_element_type_3, convert_element_type_4, mul, mul_1, rsqrt, sub, var_mean
triton_per_fused_native_layer_norm_1 = async_compile.triton('triton_per_fused_native_layer_norm_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[262144, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp53 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 / tmp10
    tmp12 = tmp2 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = tmp1 - tmp11
    tmp19 = 64.0
    tmp20 = tmp17 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp36 = tl.where(rmask, tmp34, 0)
    tmp37 = tl.sum(tmp36, 1)[:, None]
    tmp38 = tmp37 / tmp10
    tmp39 = tmp31 - tmp38
    tmp40 = tmp39 * tmp39
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK, RBLOCK])
    tmp43 = tl.where(rmask, tmp41, 0)
    tmp44 = tl.sum(tmp43, 1)[:, None]
    tmp45 = tmp30 - tmp38
    tmp46 = tmp44 / tmp19
    tmp47 = 1e-06
    tmp48 = tmp46 + tmp47
    tmp49 = libdevice.rsqrt(tmp48)
    tmp50 = tmp45 * tmp49
    tmp52 = tmp50 * tmp51
    tmp54 = tmp52 + tmp53
    tmp55 = tmp54.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp29, rmask)
    tl.store(out_ptr5 + (r1 + (64*x0)), tmp55, rmask)
''')
