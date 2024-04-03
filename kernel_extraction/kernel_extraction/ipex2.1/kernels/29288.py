

# Original file: ./AlbertForMaskedLM__0_forward_133.0/AlbertForMaskedLM__0_forward_133.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/6v/c6vlwdnwyoxyuzhhzsyqv74y4kyfeznpidu2vlhidoqpnb4m2vre.py
# Source Nodes: [add_61, add_62, l__self___predictions_decoder, l__self___predictions_layer_norm, mul_49, mul_50, mul_51, mul_52, pow_13, tanh_12], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.native_layer_norm, aten.pow, aten.tanh, aten.view]
# add_61 => add_112
# add_62 => add_113
# l__self___predictions_decoder => convert_element_type_234, view_268
# l__self___predictions_layer_norm => add_114, add_115, mul_103, mul_104, rsqrt_25, sub_38, var_mean_25
# mul_49 => mul_99
# mul_50 => mul_100
# mul_51 => mul_101
# mul_52 => mul_102
# pow_13 => convert_element_type_233, pow_13
# tanh_12 => tanh_12
triton_per_fused__to_copy_add_mul_native_layer_norm_pow_tanh_view_14 = async_compile.triton('triton_per_fused__to_copy_add_mul_native_layer_norm_pow_tanh_view_14', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mul_native_layer_norm_pow_tanh_view_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_mul_native_layer_norm_pow_tanh_view_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tmp2 * tmp1
    tmp4 = 0.044715
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 + tmp5
    tmp7 = 0.7978845608028654
    tmp8 = tmp6 * tmp7
    tmp9 = libdevice.tanh(tmp8)
    tmp10 = 0.5
    tmp11 = tmp0 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 1.0
    tmp14 = tmp9 + tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 128.0
    tmp33 = tmp31 / tmp32
    tmp34 = 1e-12
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tmp37 = tmp15 - tmp25
    tmp38 = tmp37 * tmp36
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp42.to(tl.float32)
    tl.store(out_ptr0 + (r1 + (128*x0)), tmp9, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp36, None)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp43, rmask)
    tl.store(out_ptr1 + (x0), tmp25, None)
''')
