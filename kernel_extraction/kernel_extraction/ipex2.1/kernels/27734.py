

# Original file: ./hf_T5_generate___60.0/hf_T5_generate___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/e7/ce7mndhjykxi3iwoelmfotdt7qf5ilc4umsc22ttt5rjsqt5tkpu.py
# Source Nodes: [float_2, softmax, type_as], Original ATen: [aten._softmax, aten._to_copy]
# float_2 => convert_element_type_11
# softmax => amax, div_2, exp, sub_2, sum_1
# type_as => convert_element_type_12
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
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp38 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = r2 + ((-1)*x0)
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.int64)
        tmp6 = tl.full([1, 1], 16, tl.int64)
        tmp7 = tmp5 * tmp6
        tmp8 = tmp7 + tmp3
        tmp9 = tl.abs(tmp2)
        tmp10 = tl.full([1, 1], 8, tl.int64)
        tmp11 = tmp9 < tmp10
        tmp12 = tmp9.to(tl.float32)
        tmp13 = 8.0
        tmp14 = tmp12 / tmp13
        tmp15 = tl.log(tmp14)
        tmp16 = 2.772588722239781
        tmp17 = tmp15 / tmp16
        tmp18 = tmp17 * tmp13
        tmp19 = tmp18.to(tl.int64)
        tmp20 = tmp19 + tmp10
        tmp21 = tl.full([1, 1], 15, tl.int64)
        tmp22 = triton_helpers.minimum(tmp20, tmp21)
        tmp23 = tl.where(tmp11, tmp9, tmp22)
        tmp24 = tmp8 + tmp23
        tmp25 = tl.where(tmp24 < 0, tmp24 + 32, tmp24)
        # tl.device_assert(((0 <= tmp25) & (tmp25 < 32)) | ~rmask, "index out of bounds: 0 <= tmp25 < 32")
        tmp26 = tl.load(in_ptr1 + (x1 + (8*tmp25)), rmask, eviction_policy='evict_first')
        tmp28 = tmp27.to(tl.float32)
        tmp29 = 1.0
        tmp30 = tmp29 - tmp28
        tmp31 = -3.4028234663852886e+38
        tmp32 = tmp30 * tmp31
        tmp33 = tmp26 + tmp32
        tmp34 = tmp1 + tmp33
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
        tmp39 = triton_helpers.maximum(_tmp38, tmp37)
        _tmp38 = tl.where(rmask, tmp39, _tmp38)
        tl.store(out_ptr0 + (r2 + (2048*x3)), tmp36, rmask)
    tmp38 = triton_helpers.max2(_tmp38, 1)[:, None]
    _tmp44 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp40 = tl.load(out_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp41 = tmp40 - tmp38
        tmp42 = tl.exp(tmp41)
        tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
        tmp45 = _tmp44 + tmp43
        _tmp44 = tl.where(rmask, tmp45, _tmp44)
    tmp44 = tl.sum(_tmp44, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp46 = tl.load(out_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp47 = tmp46 - tmp38
        tmp48 = tl.exp(tmp47)
        tmp49 = tmp48 / tmp44
        tmp50 = tmp49.to(tl.float32)
        tl.store(out_ptr3 + (r2 + (2048*x3)), tmp50, rmask)
''')
