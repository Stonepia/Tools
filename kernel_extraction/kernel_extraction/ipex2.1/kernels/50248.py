

# Original file: ./OPTForCausalLM__55_forward_170.13/OPTForCausalLM__55_forward_170.13_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/6b/c6bcwwl2lfmj5gsxgxg3uj7ii7cqvkkyq4xuoigeenfii42vn2qf.py
# Source Nodes: [softmax, to], Original ATen: [aten._softmax, aten._to_copy]
# softmax => amax, convert_element_type_2, div, exp, sub_1, sum_1
# to => convert_element_type_3
triton_red_fused__softmax__to_copy_3 = async_compile.triton('triton_red_fused__softmax__to_copy_3', '''
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
    size_hints=[65536, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_3(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp7 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r2 + (2048*x0) + (4194304*(x1 // 12))), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = -65504.0
        tmp4 = triton_helpers.maximum(tmp2, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = triton_helpers.maximum(_tmp7, tmp6)
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r2 + (2048*x0) + (4194304*(x1 // 12))), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = -65504.0
        tmp13 = triton_helpers.maximum(tmp11, tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14 - tmp7
        tmp16 = tl.exp(tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp20 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr1 + (r2 + (2048*x0) + (4194304*(x1 // 12))), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tmp20 + tmp21
        tmp23 = -65504.0
        tmp24 = triton_helpers.maximum(tmp22, tmp23)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp25 - tmp7
        tmp27 = tl.exp(tmp26)
        tmp28 = tmp27 / tmp18
        tmp29 = tmp28.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp28, rmask)
        tl.store(out_ptr3 + (r2 + (2048*x3)), tmp29, rmask)
''')
