

# Original file: ./DistilBertForMaskedLM__0_forward_97.0/DistilBertForMaskedLM__0_forward_97.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/fb/cfbj6p5xnl5zgjbzywfywf5gypobnt2k7o3wsjo6v7pegdhriv3y.py
# Source Nodes: [add_1, l__mod___distilbert_transformer_layer_0_ffn_lin1, l__mod___distilbert_transformer_layer_0_sa_layer_norm], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_1 => add_3
# l__mod___distilbert_transformer_layer_0_ffn_lin1 => view_19
# l__mod___distilbert_transformer_layer_0_sa_layer_norm => add_4, add_5, mul_6, mul_7, rsqrt_1, sub_2, var_mean_1
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8', '''
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
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 16384
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 768, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 768.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-12
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tmp24 / tmp20
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp25, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp29, rmask)
    tl.store(out_ptr4 + (x0), tmp30, None)
''')