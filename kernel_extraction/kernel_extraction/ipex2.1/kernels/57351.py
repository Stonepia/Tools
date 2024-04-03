

# Original file: ./BERT_pytorch___60.0/BERT_pytorch___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/kz/ckz4ofzubfq5fy7fsqme26hyt3ycv7vmzdcqegk222frjlnktmt5.py
# Source Nodes: [add_4, add_7, add_8, add_9, l__self___transformer_blocks_0_input_sublayer_dropout, l__self___transformer_blocks_0_output_sublayer_dropout, l__self___transformer_blocks_1_lambda_module_attention_linear_layers_0, mean_2, mul_2, std_2, sub_2, truediv_3], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
# add_4 => add_4
# add_7 => add_8
# add_8 => add_9
# add_9 => add_10
# l__self___transformer_blocks_0_input_sublayer_dropout => clone_6
# l__self___transformer_blocks_0_output_sublayer_dropout => clone_8
# l__self___transformer_blocks_1_lambda_module_attention_linear_layers_0 => convert_element_type_20
# mean_2 => mean_2
# mul_2 => mul_5
# std_2 => sqrt_2, var_2
# sub_2 => sub_3
# truediv_3 => div_4
triton_per_fused__to_copy_add_clone_div_mean_mul_std_sub_8 = async_compile.triton('triton_per_fused__to_copy_add_clone_div_mean_mul_std_sub_8', '''
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
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_div_mean_mul_std_sub_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_div_mean_mul_std_sub_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tmp25 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp7 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp26 = 768.0
    tmp27 = tmp10 / tmp26
    tmp28 = tmp6 - tmp27
    tmp29 = tmp25 * tmp28
    tmp30 = 767.0
    tmp31 = tmp24 / tmp30
    tmp32 = tl.sqrt(tmp31)
    tmp33 = 1e-06
    tmp34 = tmp32 + tmp33
    tmp35 = tmp29 / tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tmp37.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp38, rmask)
''')
