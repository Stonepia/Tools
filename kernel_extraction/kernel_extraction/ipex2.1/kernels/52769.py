

# Original file: ./AlbertForMaskedLM__0_forward_205.0/AlbertForMaskedLM__0_forward_205.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/ai/caixquc4cmszoiw7xpjt62qjpedq3a2ywsioxy77g6tlbuqr4zis.py
# Source Nodes: [add_61, add_62, l__mod___predictions_decoder, l__mod___predictions_layer_norm, mul_49, mul_50, mul_51, mul_52, pow_13, tanh_12], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.pow, aten.tanh, aten.view]
# add_61 => add_112
# add_62 => add_113
# l__mod___predictions_decoder => view_268
# l__mod___predictions_layer_norm => add_114, add_115, mul_103, mul_104, rsqrt_25, sub_38, var_mean_25
# mul_49 => mul_99
# mul_50 => mul_100
# mul_51 => mul_101
# mul_52 => mul_102
# pow_13 => pow_13
# tanh_12 => tanh_12
triton_per_fused_add_mul_native_layer_norm_pow_tanh_view_10 = async_compile.triton('triton_per_fused_add_mul_native_layer_norm_pow_tanh_view_10', '''
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_pow_tanh_view_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_mul_native_layer_norm_pow_tanh_view_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0)
    tmp37 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 * tmp0
    tmp2 = tmp1 * tmp0
    tmp3 = 0.044715
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 + tmp4
    tmp6 = 0.7978845608028654
    tmp7 = tmp5 * tmp6
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = 0.5
    tmp10 = tmp0 * tmp9
    tmp11 = 1.0
    tmp12 = tmp8 + tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
    tmp28 = tl.where(rmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = 128.0
    tmp31 = tmp29 / tmp30
    tmp32 = 1e-12
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp13 - tmp23
    tmp36 = tmp35 * tmp34
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tl.store(out_ptr0 + (r1 + (128*x0)), tmp8, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp34, None)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp40, rmask)
    tl.store(out_ptr1 + (x0), tmp23, None)
''')
