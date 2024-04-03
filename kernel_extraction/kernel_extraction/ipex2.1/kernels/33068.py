

# Original file: ./MobileBertForQuestionAnswering__0_forward_205.0/MobileBertForQuestionAnswering__0_forward_205.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/zj/czjxi67jg2gfdlsmosk2imi3x2tjsdzuhayxneww5432glfl2gka.py
# Source Nodes: [add_5, l__mod___mobilebert_encoder_layer_0_attention_self_dropout, softmax], Original ATen: [aten._softmax, aten.add, aten.native_dropout]
# add_5 => div
# l__mod___mobilebert_encoder_layer_0_attention_self_dropout => gt, mul_4, mul_5
# softmax => amax, convert_element_type_1, convert_element_type_2, div_1, exp, sub_1, sum_1
triton_per_fused__softmax_add_native_dropout_7 = async_compile.triton('triton_per_fused__softmax_add_native_dropout_7', '''
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
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*i64', 2: '*i1', 3: '*bf16', 4: '*bf16', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_native_dropout_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_add_native_dropout_7(in_ptr0, in_ptr1, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
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
    tmp1 = 5.656854249492381
    tmp2 = tmp0 / tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.max2(tmp6, 1)[:, None]
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tl.load(in_ptr1 + load_seed_offset)
    tmp15 = r1 + (128*x0)
    tmp16 = tl.rand(tmp14, (tmp15).to(tl.uint32))
    tmp17 = tmp16.to(tl.float32)
    tmp18 = 0.1
    tmp19 = tmp17 > tmp18
    tmp20 = tmp9 / tmp13
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19.to(tl.float32)
    tmp23 = tmp22 * tmp21
    tmp24 = 1.1111111111111112
    tmp25 = tmp23 * tmp24
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp19, rmask)
    tl.store(out_ptr4 + (r1 + (128*x0)), tmp21, rmask)
    tl.store(out_ptr5 + (r1 + (128*x0)), tmp25, rmask)
''')
