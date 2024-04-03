

# Original file: ./BERT_pytorch___60.0/BERT_pytorch___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/jp/cjpixolwq5gykgi3tyv4p4v6cmpxzqg4c5l3xrdmlmrbwgfdcr7s.py
# Source Nodes: [add_4, add_5, add_6, l__self___transformer_blocks_0_feed_forward_w_1, l__self___transformer_blocks_0_input_sublayer_dropout, mean_1, mul_1, std_1, sub_1, truediv_2], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
# add_4 => add_4
# add_5 => add_5
# add_6 => add_6
# l__self___transformer_blocks_0_feed_forward_w_1 => convert_element_type_13
# l__self___transformer_blocks_0_input_sublayer_dropout => clone_6
# mean_1 => mean_1
# mul_1 => mul_1
# std_1 => sqrt_1, var_1
# sub_1 => sub_2
# truediv_2 => div_3
triton_per_fused__to_copy_add_clone_div_mean_mul_std_sub_6 = async_compile.triton('triton_per_fused__to_copy_add_clone_div_mean_mul_std_sub_6', '''
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
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_div_mean_mul_std_sub_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_div_mean_mul_std_sub_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
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
    tmp22 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tl.full([1], 768, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 / tmp14
    tmp16 = tmp4 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp23 = 768.0
    tmp24 = tmp7 / tmp23
    tmp25 = tmp3 - tmp24
    tmp26 = tmp22 * tmp25
    tmp27 = 767.0
    tmp28 = tmp21 / tmp27
    tmp29 = tl.sqrt(tmp28)
    tmp30 = 1e-06
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp31
    tmp34 = tmp32 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp35, rmask)
''')
