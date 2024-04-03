

# Original file: ./AlbertForQuestionAnswering__0_forward_133.0/AlbertForQuestionAnswering__0_forward_133.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/gq/cgqt2ncmt5csgqmd6ww7p4rrv2u6b6e2aarjn7sobmjn3sipqdyb.py
# Source Nodes: [add_1, l__self___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_attention_dropout, matmul_1, softmax, truediv], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.clone, aten.div]
# add_1 => convert_element_type_default
# l__self___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_attention_dropout => clone_3
# matmul_1 => convert_element_type_9
# softmax => amax, div_1, exp, sub_2, sum_1
# truediv => div
triton_per_fused__softmax__to_copy_add_clone_div_6 = async_compile.triton('triton_per_fused__softmax__to_copy_add_clone_div_6', '''
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
    size_hints=[131072, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_add_clone_div_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_add_clone_div_6(in_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 131072
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tmp9 / tmp13
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp14, rmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp15, rmask)
''')
