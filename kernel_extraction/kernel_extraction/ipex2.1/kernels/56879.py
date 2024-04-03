

# Original file: ./twins_pcpvt_base___60.0/twins_pcpvt_base___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/iy/ciy3neg6mizsw64uu6oocjrjjeclyh2dmsosdhev52lw5kfkub5u.py
# Source Nodes: [l__mod___blocks_1_0_norm1, l__mod___patch_embeds_1_norm], Original ATen: [aten.native_layer_norm]
# l__mod___blocks_1_0_norm1 => add_32, add_33, convert_element_type_28, convert_element_type_29, mul_31, mul_32, rsqrt_11, sub_11, var_mean_11
# l__mod___patch_embeds_1_norm => add_30, add_31, clone_12, convert_element_type_26, convert_element_type_27, mul_29, mul_30, rsqrt_10, sub_10, var_mean_10
triton_per_fused_native_layer_norm_13 = async_compile.triton('triton_per_fused_native_layer_norm_13', '''
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
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_native_layer_norm_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp53 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp56 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 / tmp10
    tmp12 = tmp2 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = tmp1 - tmp11
    tmp19 = 128.0
    tmp20 = tmp17 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 * tmp26
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.where(rmask & xmask, tmp33, 0)
    tmp36 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
    tmp38 = tl.where(rmask & xmask, tmp36, 0)
    tmp39 = tl.sum(tmp38, 1)[:, None]
    tmp40 = tmp39 / tmp10
    tmp41 = tmp33 - tmp40
    tmp42 = tmp41 * tmp41
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
    tmp45 = tl.where(rmask & xmask, tmp43, 0)
    tmp46 = tl.sum(tmp45, 1)[:, None]
    tmp47 = tmp32 - tmp40
    tmp48 = tmp46 / tmp19
    tmp49 = 1e-06
    tmp50 = tmp48 + tmp49
    tmp51 = libdevice.rsqrt(tmp50)
    tmp52 = tmp47 * tmp51
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tmp52 * tmp54
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp55 + tmp57
    tmp59 = tmp58.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (128*x0)), tmp59, rmask & xmask)
''')
