

# Original file: ./T5ForConditionalGeneration__0_backward_171.1/T5ForConditionalGeneration__0_backward_171.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/ss/csswqukgkudgdrovcvlpfsy4bzcd4v7ene3oj5cfzthfevvsehnr.py
# Source Nodes: [mul_69, to_70], Original ATen: [aten._to_copy, aten.mul, aten.native_dropout_backward, aten.sum]
# mul_69 => mul_195
# to_70 => convert_element_type_108
triton_red_fused__to_copy_mul_native_dropout_backward_sum_3 = async_compile.triton('triton_red_fused__to_copy_mul_native_dropout_backward_sum_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*i1', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_native_dropout_backward_sum_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_mul_native_dropout_backward_sum_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr3 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.04419417382415922
        tmp2 = tmp0 * tmp1
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 1.1111111111111112
        tmp6 = tmp4 * tmp5
        tmp7 = tmp2 * tmp6
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 * tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp7 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, None)
''')