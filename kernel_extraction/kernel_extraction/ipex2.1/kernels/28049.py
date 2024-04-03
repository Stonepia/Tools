

# Original file: ./BartForConditionalGeneration__100_forward_309.24/BartForConditionalGeneration__100_forward_309.24.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/vd/cvdol6mekcbqxz65wykamv46yhreitccd4prrop4aabod3o6zeq4.py
# Source Nodes: [add_1, dropout_1, l__self___encoder_attn_q_proj, l__self___self_attn_layer_norm], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_1 => add_1
# dropout_1 => gt, mul_1, mul_2
# l__self___encoder_attn_q_proj => view_18
# l__self___self_attn_layer_norm => add_2, add_3, mul_3, mul_4, rsqrt, sub_1, var_mean
triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_4 = async_compile.triton('triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_4', '''
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
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp5 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp35 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (1024*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp6 = tmp4.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 + tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tl.full([1], 1024, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 / tmp20
    tmp22 = tmp12 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = tmp11 - tmp21
    tmp29 = 1024.0
    tmp30 = tmp27 / tmp29
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp28 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tmp33 / tmp29
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp4, rmask)
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp34, rmask)
    tl.store(out_ptr5 + (r1 + (1024*x0)), tmp38, rmask)
    tl.store(out_ptr6 + (x0), tmp39, None)
''')
