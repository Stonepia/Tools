

# Original file: ./tnt_s_patch16_224___60.0/tnt_s_patch16_224___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/vy/cvy6lhaxjhrxziofjuvbykdilhxp3c7igxq4y7mm52iyri3gmz32.py
# Source Nodes: [add_8, l__mod___blocks_1_mlp_in_drop2, l__mod___blocks_1_norm1_proj, l__mod___blocks_2_norm_in], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_8 => add_31
# l__mod___blocks_1_mlp_in_drop2 => clone_26
# l__mod___blocks_1_norm1_proj => add_32, add_33, clone_27, convert_element_type_32, convert_element_type_33, mul_30, mul_31, rsqrt_9, sub_12, var_mean_9
# l__mod___blocks_2_norm_in => add_43, convert_element_type_43, mul_41
triton_per_fused_add_clone_native_layer_norm_23 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_23', '''
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
    size_hints=[524288, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (24*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (24*x0)), rmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp37 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 / tmp12
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = tmp3 - tmp13
    tmp21 = 24.0
    tmp22 = tmp19 / tmp21
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 * tmp28
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp26 * tmp35
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp36 + tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (24*x0)), tmp33, rmask)
    tl.store(out_ptr3 + (r1 + (24*x0)), tmp40, rmask)
''')
