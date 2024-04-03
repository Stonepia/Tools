

# Original file: ./hf_T5___60.0/hf_T5___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/uc/cuciczpphxmj2ly2nnuu4ortiixhev5crfccsiy55wpa4bz6623w.py
# Source Nodes: [float_9, softmax_6, type_as_6], Original ATen: [aten._softmax, aten._to_copy]
# float_9 => convert_element_type_104
# softmax_6 => amax_6, div_10, exp_6, sub_11, sum_7
# type_as_6 => convert_element_type_105
triton_red_fused__softmax__to_copy_2 = async_compile.triton('triton_red_fused__softmax__to_copy_2', '''
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
    size_hints=[16384, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_2(in_ptr0, in_ptr1, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp34 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = (-1)*(tl.minimum(0, r2 + ((-1)*x0), tl.PropagateNan.NONE))
        tmp3 = tl.full([1, 1], 16, tl.int64)
        tmp4 = tmp2 < tmp3
        tmp5 = tmp2.to(tl.float32)
        tmp6 = 16.0
        tmp7 = tmp5 / tmp6
        tmp8 = tl.log(tmp7)
        tmp9 = 2.0794415416798357
        tmp10 = tmp8 / tmp9
        tmp11 = tmp10 * tmp6
        tmp12 = tmp11.to(tl.int64)
        tmp13 = tmp12 + tmp3
        tmp14 = tl.full([1, 1], 31, tl.int64)
        tmp15 = triton_helpers.minimum(tmp13, tmp14)
        tmp16 = tl.where(tmp4, tmp2, tmp15)
        tmp17 = tl.full([1, 1], 0, tl.int64)
        tmp18 = tmp16 + tmp17
        tmp19 = tl.where(tmp18 < 0, tmp18 + 32, tmp18)
        # tl.device_assert((0 <= tmp19) & (tmp19 < 32), "index out of bounds: 0 <= tmp19 < 32")
        tmp20 = tl.load(in_ptr1 + (x1 + (8*tmp19)), None, eviction_policy='evict_first')
        tmp21 = r2
        tmp22 = x0
        tmp23 = tmp21 <= tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp25 = 1.0
        tmp26 = tmp25 - tmp24
        tmp27 = -3.4028234663852886e+38
        tmp28 = tmp26 * tmp27
        tmp29 = tmp20 + tmp28
        tmp30 = tmp1 + tmp29
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = triton_helpers.maximum(_tmp34, tmp33)
        _tmp34 = tl.where(rmask, tmp35, _tmp34)
        tl.store(out_ptr0 + (r2 + (2048*x3)), tmp32, rmask)
    tmp34 = triton_helpers.max2(_tmp34, 1)[:, None]
    _tmp40 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp36 = tl.load(out_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp37 = tmp36 - tmp34
        tmp38 = tl.exp(tmp37)
        tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
        tmp41 = _tmp40 + tmp39
        _tmp40 = tl.where(rmask, tmp41, _tmp40)
    tmp40 = tl.sum(_tmp40, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp42 = tl.load(out_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp43 = tmp42 - tmp34
        tmp44 = tl.exp(tmp43)
        tmp45 = tmp44 / tmp40
        tmp46 = tmp45.to(tl.float32)
        tl.store(out_ptr3 + (r2 + (2048*x3)), tmp46, rmask)
''')
