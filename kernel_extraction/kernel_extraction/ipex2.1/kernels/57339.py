

# Original file: ./BERT_pytorch___60.0/BERT_pytorch___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/xj/cxjatead262fuyfjllauxquvyichekfm7ojjncay6oc4j63w3xqi.py
# Source Nodes: [add_4, add_7, add_8, add_9, l__mod___transformer_blocks_0_input_sublayer_dropout, l__mod___transformer_blocks_0_output_sublayer_dropout, mean_2, mul_2, std_2, sub_2, truediv_3], Original ATen: [aten.add, aten.clone, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
# add_4 => add_4
# add_7 => add_8
# add_8 => add_9
# add_9 => add_10
# l__mod___transformer_blocks_0_input_sublayer_dropout => clone_6
# l__mod___transformer_blocks_0_output_sublayer_dropout => clone_8
# mean_2 => mean_2
# mul_2 => mul_5
# std_2 => sqrt_2, var_2
# sub_2 => sub_3
# truediv_3 => div_4
triton_per_fused_add_clone_div_mean_mul_std_sub_8 = async_compile.triton('triton_per_fused_add_clone_div_mean_mul_std_sub_8', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_div_mean_mul_std_sub_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_div_mean_mul_std_sub_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 768, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp5 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = 768.0
    tmp25 = tmp8 / tmp24
    tmp26 = tmp4 - tmp25
    tmp27 = tmp23 * tmp26
    tmp28 = 767.0
    tmp29 = tmp22 / tmp28
    tmp30 = tl.sqrt(tmp29)
    tmp31 = 1e-06
    tmp32 = tmp30 + tmp31
    tmp33 = tmp27 / tmp32
    tmp35 = tmp33 + tmp34
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp35, rmask)
''')
