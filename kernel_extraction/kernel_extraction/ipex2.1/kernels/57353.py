

# Original file: ./BERT_pytorch___60.0/BERT_pytorch___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/bt/cbtmp3xhh76fplinmsg6zs6ahfhlduttw2p5tgi6aat7o7cqmdzs.py
# Source Nodes: [add_10, add_13, add_14, add_15, add_4, add_7, l__self___transformer_blocks_0_input_sublayer_dropout, l__self___transformer_blocks_0_output_sublayer_dropout, l__self___transformer_blocks_1_input_sublayer_dropout, l__self___transformer_blocks_1_output_sublayer_dropout, l__self___transformer_blocks_2_lambda_module_attention_linear_layers_0, mean_4, mul_4, std_4, sub_4, truediv_6], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
# add_10 => add_11
# add_13 => add_15
# add_14 => add_16
# add_15 => add_17
# add_4 => add_4
# add_7 => add_8
# l__self___transformer_blocks_0_input_sublayer_dropout => clone_6
# l__self___transformer_blocks_0_output_sublayer_dropout => clone_8
# l__self___transformer_blocks_1_input_sublayer_dropout => clone_15
# l__self___transformer_blocks_1_output_sublayer_dropout => clone_17
# l__self___transformer_blocks_2_lambda_module_attention_linear_layers_0 => convert_element_type_40
# mean_4 => mean_4
# mul_4 => mul_10
# std_4 => sqrt_4, var_4
# sub_4 => sub_6
# truediv_6 => div_8
triton_per_fused__to_copy_add_clone_div_mean_mul_std_sub_10 = async_compile.triton('triton_per_fused__to_copy_add_clone_div_mean_mul_std_sub_10', '''
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
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*bf16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_div_mean_mul_std_sub_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_div_mean_mul_std_sub_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel):
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
    tmp10 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tl.full([1], 768, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp25 = tmp13 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp32 = 768.0
    tmp33 = tmp16 / tmp32
    tmp34 = tmp12 - tmp33
    tmp35 = tmp31 * tmp34
    tmp36 = 767.0
    tmp37 = tmp30 / tmp36
    tmp38 = tl.sqrt(tmp37)
    tmp39 = 1e-06
    tmp40 = tmp38 + tmp39
    tmp41 = tmp35 / tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp43.to(tl.float32)
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp12, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp44, rmask)
''')
