

# Original file: ./hf_T5_generate__44_inference_84.24/hf_T5_generate__44_inference_84.24_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/gq/cgqnyqxpz2m2dnwaox6uxpsutvsmtnscvmsrmc4v5ipft52ftn6g.py
# Source Nodes: [float_3, softmax_1, type_as_1], Original ATen: [aten._softmax, aten._to_copy]
# float_3 => convert_element_type_13
# softmax_1 => amax_1, div_3, exp_1, sub_9, sum_2
# type_as_1 => convert_element_type_14
triton_red_fused__softmax__to_copy_8 = async_compile.triton('triton_red_fused__softmax__to_copy_8', '''
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
    size_hints=[8, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_8(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = 1.0
        tmp4 = tmp3 - tmp2
        tmp5 = -65504.0
        tmp6 = tmp4 * tmp5
        tmp7 = 0.0
        tmp8 = tmp7 + tmp6
        tmp9 = tmp0 + tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = triton_helpers.maximum(_tmp12, tmp11)
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = triton_helpers.max2(_tmp12, 1)[:, None]
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 1.0
        tmp18 = tmp17 - tmp16
        tmp19 = -65504.0
        tmp20 = tmp18 * tmp19
        tmp21 = 0.0
        tmp22 = tmp21 + tmp20
        tmp23 = tmp14 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp24 - tmp12
        tmp26 = tl.exp(tmp25)
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tmp31.to(tl.float32)
        tmp33 = 1.0
        tmp34 = tmp33 - tmp32
        tmp35 = -65504.0
        tmp36 = tmp34 * tmp35
        tmp37 = 0.0
        tmp38 = tmp37 + tmp36
        tmp39 = tmp30 + tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp41 = tmp40 - tmp12
        tmp42 = tl.exp(tmp41)
        tmp43 = tmp42 / tmp28
        tmp44 = tmp43.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp44, rmask & xmask)
''')
