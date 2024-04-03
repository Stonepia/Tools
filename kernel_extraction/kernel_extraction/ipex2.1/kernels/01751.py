

# Original file: ./hf_T5_generate__27_inference_67.7/hf_T5_generate__27_inference_67.7.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/kz/ckzr2b62w6gwswnvhb74gu7pggj5psyuwa2klolykxxi4xwpggs2.py
# Source Nodes: [float_2, softmax, type_as], Original ATen: [aten._softmax, aten._to_copy]
# float_2 => convert_element_type_11
# softmax => amax, div_2, exp, sub_8, sum_1
# type_as => convert_element_type_12
triton_red_fused__softmax__to_copy_6 = async_compile.triton('triton_red_fused__softmax__to_copy_6', '''
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
    size_hints=[8, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_6(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr3, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp32 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + x0 + (ks0*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = (-1)*(tl.minimum(0, r1 + ((-1)*ks0), tl.PropagateNan.NONE))
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
        tmp20 = tl.load(in_ptr1 + (x0 + (8*tmp19)), xmask, eviction_policy='evict_last')
        tmp22 = 1.0
        tmp23 = tmp21 * tmp22
        tmp24 = tmp22 - tmp23
        tmp25 = -3.4028234663852886e+38
        tmp26 = tmp24 * tmp25
        tmp27 = tmp20 + tmp26
        tmp28 = tmp1 + tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = triton_helpers.maximum(_tmp32, tmp31)
        _tmp32 = tl.where(rmask & xmask, tmp33, _tmp32)
    tmp32 = triton_helpers.max2(_tmp32, 1)[:, None]
    _tmp68 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp34 = tl.load(in_ptr0 + (r1 + x0 + (ks0*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp55 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tmp34.to(tl.float32)
        tmp36 = (-1)*(tl.minimum(0, r1 + ((-1)*ks0), tl.PropagateNan.NONE))
        tmp37 = tl.full([1, 1], 16, tl.int64)
        tmp38 = tmp36 < tmp37
        tmp39 = tmp36.to(tl.float32)
        tmp40 = 16.0
        tmp41 = tmp39 / tmp40
        tmp42 = tl.log(tmp41)
        tmp43 = 2.0794415416798357
        tmp44 = tmp42 / tmp43
        tmp45 = tmp44 * tmp40
        tmp46 = tmp45.to(tl.int64)
        tmp47 = tmp46 + tmp37
        tmp48 = tl.full([1, 1], 31, tl.int64)
        tmp49 = triton_helpers.minimum(tmp47, tmp48)
        tmp50 = tl.where(tmp38, tmp36, tmp49)
        tmp51 = tl.full([1, 1], 0, tl.int64)
        tmp52 = tmp50 + tmp51
        tmp53 = tl.where(tmp52 < 0, tmp52 + 32, tmp52)
        # tl.device_assert((0 <= tmp53) & (tmp53 < 32), "index out of bounds: 0 <= tmp53 < 32")
        tmp54 = tl.load(in_ptr1 + (x0 + (8*tmp53)), xmask, eviction_policy='evict_first')
        tmp56 = 1.0
        tmp57 = tmp55 * tmp56
        tmp58 = tmp56 - tmp57
        tmp59 = -3.4028234663852886e+38
        tmp60 = tmp58 * tmp59
        tmp61 = tmp54 + tmp60
        tmp62 = tmp35 + tmp61
        tmp63 = tmp62.to(tl.float32)
        tmp64 = tmp63.to(tl.float32)
        tmp65 = tmp64 - tmp32
        tmp66 = tl.exp(tmp65)
        tmp67 = tl.broadcast_to(tmp66, [XBLOCK, RBLOCK])
        tmp69 = _tmp68 + tmp67
        _tmp68 = tl.where(rmask & xmask, tmp69, _tmp68)
        tl.store(out_ptr1 + (r1 + x0 + (ks0*x0)), tmp66, rmask & xmask)
    tmp68 = tl.sum(_tmp68, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp70 = tl.load(out_ptr1 + (r1 + x0 + (ks0*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp71 = tmp70 / tmp68
        tmp72 = tmp71.to(tl.float32)
        tl.store(out_ptr3 + (r1 + x0 + (ks0*x0)), tmp72, rmask & xmask)
''')
