

# Original file: ./AlbertForMaskedLM__0_backward_135.1/AlbertForMaskedLM__0_backward_135.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/5k/c5kkl4qtb62zfdelgwy33r4f6ym376vf4eqo4fpl6jmon2jx5c7f.py
# Source Nodes: [add_62, l__mod___predictions_layer_norm, mul_49, mul_52], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.pow, aten.tanh_backward]
# add_62 => add_113
# l__mod___predictions_layer_norm => convert_element_type_75
# mul_49 => mul_99
# mul_52 => mul_102
triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_4 = async_compile.triton('triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_4', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp10 = 0.5
    tmp11 = tmp9 * tmp10
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 - tmp17
    tmp20 = tmp18 * tmp19
    tmp21 = tmp4 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = 128.0
    tmp27 = tmp19 / tmp26
    tmp28 = tmp4 * tmp26
    tmp29 = tmp28 - tmp8
    tmp30 = tmp20 * tmp25
    tmp31 = tmp29 - tmp30
    tmp32 = tmp27 * tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33 * tmp11
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp12.to(tl.float32)
    tmp37 = tmp36 * tmp36
    tmp38 = tmp13 - tmp37
    tmp39 = tmp35 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tmp41 = 0.7978845608028654
    tmp42 = tmp40 * tmp41
    tmp43 = 0.044715
    tmp44 = tmp42 * tmp43
    tmp45 = tmp9 * tmp9
    tmp46 = 3.0
    tmp47 = tmp45 * tmp46
    tmp48 = tmp44 * tmp47
    tmp49 = tmp42 + tmp48
    tmp50 = tmp33 * tmp14
    tmp51 = tmp50 * tmp10
    tmp52 = tmp49 + tmp51
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp52, rmask)
''')
