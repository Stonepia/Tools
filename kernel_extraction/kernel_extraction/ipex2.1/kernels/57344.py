

# Original file: ./BERT_pytorch___60.0/BERT_pytorch___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/23/c232i3lgnpcfqkpqqrdgvi6evingy2e7ew37a3y5ukeylppotzfe.py
# Source Nodes: [add, add_1, add_2, add_3, forward, forward_1, l__self___transformer_blocks_0_lambda_module_attention_linear_layers_0, mean, mul, std, sub, truediv], Original ATen: [aten._to_copy, aten.add, aten.div, aten.embedding, aten.mean, aten.mul, aten.std, aten.sub]
# add => add
# add_1 => add_1
# add_2 => add_2
# add_3 => add_3
# forward => embedding
# forward_1 => embedding_1
# l__self___transformer_blocks_0_lambda_module_attention_linear_layers_0 => convert_element_type
# mean => mean
# mul => mul
# std => sqrt, var
# sub => sub
# truediv => div
triton_per_fused__to_copy_add_div_embedding_mean_mul_std_sub_1 = async_compile.triton('triton_per_fused__to_copy_add_div_embedding_mean_mul_std_sub_1', '''
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
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*bf16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_embedding_mean_mul_std_sub_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_embedding_mean_mul_std_sub_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel):
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
    x3 = xindex
    r2 = rindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 20005, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 20005), "index out of bounds: 0 <= tmp1 < 20005")
    tmp2 = tl.load(in_ptr1 + (r2 + (768*tmp1)), rmask, other=0.0)
    tmp4 = tmp2 + tmp3
    tmp6 = tl.where(tmp5 < 0, tmp5 + 3, tmp5)
    # tl.device_assert((0 <= tmp6) & (tmp6 < 3), "index out of bounds: 0 <= tmp6 < 3")
    tmp7 = tl.load(in_ptr4 + (r2 + (768*tmp6)), rmask, other=0.0)
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 768, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp9 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp28 = 768.0
    tmp29 = tmp12 / tmp28
    tmp30 = tmp8 - tmp29
    tmp31 = tmp27 * tmp30
    tmp32 = 767.0
    tmp33 = tmp26 / tmp32
    tmp34 = tl.sqrt(tmp33)
    tmp35 = 1e-06
    tmp36 = tmp34 + tmp35
    tmp37 = tmp31 / tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp8, rmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp40, rmask)
''')
