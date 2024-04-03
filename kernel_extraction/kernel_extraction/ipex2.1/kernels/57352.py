

# Original file: ./BERT_pytorch___60.0/BERT_pytorch___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/ck/cckv55gswptwywbwddum6m6cphvcgl7asnio4dphoefn66qxnsx3.py
# Source Nodes: [add_10, add_11, add_12, add_4, add_7, l__self___transformer_blocks_0_input_sublayer_dropout, l__self___transformer_blocks_0_output_sublayer_dropout, l__self___transformer_blocks_1_feed_forward_w_1, l__self___transformer_blocks_1_input_sublayer_dropout, mean_3, mul_3, std_3, sub_3, truediv_5], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
# add_10 => add_11
# add_11 => add_12
# add_12 => add_13
# add_4 => add_4
# add_7 => add_8
# l__self___transformer_blocks_0_input_sublayer_dropout => clone_6
# l__self___transformer_blocks_0_output_sublayer_dropout => clone_8
# l__self___transformer_blocks_1_feed_forward_w_1 => convert_element_type_33
# l__self___transformer_blocks_1_input_sublayer_dropout => clone_15
# mean_3 => mean_3
# mul_3 => mul_6
# std_3 => sqrt_3, var_3
# sub_3 => sub_5
# truediv_5 => div_7
triton_per_fused__to_copy_add_clone_div_mean_mul_std_sub_9 = async_compile.triton('triton_per_fused__to_copy_add_clone_div_mean_mul_std_sub_9', '''
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
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_div_mean_mul_std_sub_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_div_mean_mul_std_sub_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 2048
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
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tl.full([1], 768, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 / tmp20
    tmp22 = tmp10 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp29 = 768.0
    tmp30 = tmp13 / tmp29
    tmp31 = tmp9 - tmp30
    tmp32 = tmp28 * tmp31
    tmp33 = 767.0
    tmp34 = tmp27 / tmp33
    tmp35 = tl.sqrt(tmp34)
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tmp32 / tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = tmp40.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp41, rmask)
''')
