

# Original file: ./MT5ForConditionalGeneration__0_forward_205.0/MT5ForConditionalGeneration__0_forward_205.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/bs/cbsdazujcy6mrraty24ej4frzq2hfe276ksrvaenkyc5h22beqxf.py
# Source Nodes: [dropout_9, float_12, softmax_9, type_as_9], Original ATen: [aten._softmax, aten._to_copy, aten.native_dropout]
# dropout_9 => gt_38, mul_159, mul_160
# float_12 => convert_element_type_160
# softmax_9 => amax_9, div_13, exp_9, sub_14, sum_10
# type_as_9 => convert_element_type_161
triton_per_fused__softmax__to_copy_native_dropout_19 = async_compile.triton('triton_per_fused__softmax__to_copy_native_dropout_19', '''
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*i64', 2: '*i1', 3: '*fp32', 4: '*bf16', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_native_dropout_19', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_native_dropout_19(in_ptr0, in_ptr1, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
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
    tmp19 = tmp17.to(tl.float32)
    tmp20 = tmp18.to(tl.float32)
    tmp21 = tmp19 * tmp20
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp17, rmask)
    tl.store(out_ptr4 + (r1 + (128*x0)), tmp18, rmask)
    tl.store(out_ptr5 + (r1 + (128*x0)), tmp23, rmask)
''')
