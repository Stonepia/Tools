

# Original file: ./BERT_pytorch___60.0/BERT_pytorch___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/qr/cqrhxdtls365jufn76auuzrq5c7mcupd7sbrsgpcjkoxma4y7clh.py
# Source Nodes: [add_4, add_7, add_8, add_9, l__mod___transformer_blocks_0_input_sublayer_dropout, l__mod___transformer_blocks_0_output_sublayer_dropout, mean_2, mul_2, std_2, sub_2, truediv_3], Original ATen: [aten.add, aten.clone, aten.div, aten.mean, aten.mul, aten.std, aten.sub]
# add_4 => add_4
# add_7 => add_8
# add_8 => add_9
# add_9 => add_10
# l__mod___transformer_blocks_0_input_sublayer_dropout => clone_6
# l__mod___transformer_blocks_0_output_sublayer_dropout => clone_8
# mean_2 => mean_2
# mul_2 => mul_5
# std_2 => convert_element_type_8, convert_element_type_9, sqrt_2, var_2
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_div_mean_mul_std_sub_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp37 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp11 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp15 = tl.full([1], 768, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 / tmp16
    tmp18 = tmp6 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp25 = 768.0
    tmp26 = tmp9 / tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp4 - tmp27
    tmp29 = tmp24 * tmp28
    tmp30 = 767.0
    tmp31 = tmp23 / tmp30
    tmp32 = tl.sqrt(tmp31)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = 1e-06
    tmp35 = tmp33 + tmp34
    tmp36 = tmp29 / tmp35
    tmp38 = tmp36 + tmp37
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp38, rmask)
''')
