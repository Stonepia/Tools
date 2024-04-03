

# Original file: ./AlbertForMaskedLM__0_backward_135.1/AlbertForMaskedLM__0_backward_135.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/6x/c6xht5h2hzntkadhch47acsecyoyvsta4zi3lugbmdz5mzfvjaic.py
# Source Nodes: [add_62, l__self___predictions_layer_norm, mul_49, mul_52, pow_13], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.pow, aten.tanh_backward]
# add_62 => add_113
# l__self___predictions_layer_norm => mul_103, sub_38
# mul_49 => mul_99
# mul_52 => mul_102
# pow_13 => convert_element_type_233
triton_per_fused__to_copy_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_5 = async_compile.triton('triton_per_fused__to_copy_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_5', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (r1 + (128*x0)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 * tmp14
    tmp17 = tmp15 - tmp16
    tmp19 = tmp17 * tmp18
    tmp20 = tmp3 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = 128.0
    tmp26 = tmp18 / tmp25
    tmp27 = tmp3 * tmp25
    tmp28 = tmp27 - tmp7
    tmp29 = tmp19 * tmp24
    tmp30 = tmp28 - tmp29
    tmp31 = tmp26 * tmp30
    tmp32 = tmp31 * tmp11
    tmp33 = tmp12 * tmp12
    tmp34 = tmp13 - tmp33
    tmp35 = tmp32 * tmp34
    tmp36 = 0.7978845608028654
    tmp37 = tmp35 * tmp36
    tmp38 = tmp37.to(tl.float32)
    tmp39 = 0.044715
    tmp40 = tmp37 * tmp39
    tmp41 = tmp8.to(tl.float32)
    tmp42 = tmp41 * tmp41
    tmp43 = 3.0
    tmp44 = tmp42 * tmp43
    tmp45 = tmp40 * tmp44
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp38 + tmp46
    tmp48 = tmp31 * tmp14
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp49 * tmp9
    tmp51 = tmp47 + tmp50
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp51, rmask)
''')
