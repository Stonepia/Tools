

# Original file: ./M2M100ForConditionalGeneration__62_forward_193.16/M2M100ForConditionalGeneration__62_forward_193.16.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/w7/cw7hba4tgplk7y5owo5va5wdjg66wttxsnv663izfzzbli5zigp2.py
# Source Nodes: [dropout_2, softmax_1], Original ATen: [aten._softmax, aten.native_dropout]
# dropout_2 => gt_2, mul_10, mul_11
# softmax_1 => amax_1, convert_element_type_6, convert_element_type_7, div_1, exp_1, sub_3, sum_2
triton_per_fused__softmax_native_dropout_6 = async_compile.triton('triton_per_fused__softmax_native_dropout_6', '''
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i64', 2: '*i1', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_native_dropout_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_native_dropout_6(in_ptr0, in_ptr1, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
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
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask, tmp2, float("-inf"))
    tmp5 = triton_helpers.max2(tmp4, 1)[:, None]
    tmp6 = tmp1 - tmp5
    tmp7 = tl.exp(tmp6)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.load(in_ptr1 + load_seed_offset)
    tmp13 = r1 + (128*x0)
    tmp14 = tl.rand(tmp12, (tmp13).to(tl.uint32))
    tmp15 = tmp14.to(tl.float32)
    tmp16 = 0.1
    tmp17 = tmp15 > tmp16
    tmp18 = tmp7 / tmp11
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17.to(tl.float32)
    tmp21 = tmp20 * tmp19
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp17, rmask)
    tl.store(out_ptr4 + (r1 + (128*x0)), tmp19, rmask)
    tl.store(out_ptr5 + (r1 + (128*x0)), tmp23, rmask)
''')