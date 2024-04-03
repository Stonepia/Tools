

# Original file: ./cm3leon_generate__28_inference_68.8/cm3leon_generate__28_inference_68.8.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/qe/cqeqljraxcebkjwzikfgufhw2iy5yppknw47yhayks4unbu7p7mb.py
# Source Nodes: [iadd_2, nan_to_num, nan_to_num_1, softmax_1, type_as_3], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.nan_to_num]
# iadd_2 => add_11
# nan_to_num => full_default_2, full_default_3, full_default_4
# nan_to_num_1 => convert_element_type_14, eq_6, eq_7, isnan_1, where_4, where_5, where_6
# softmax_1 => amax_1, convert_element_type_16, div_1, exp_1, sub_5, sum_2
# type_as_3 => convert_element_type_17
triton_red_fused__softmax__to_copy_add_nan_to_num_12 = async_compile.triton('triton_red_fused__softmax__to_copy_add_nan_to_num_12', '''
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
    size_hints=[16, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_add_nan_to_num_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_add_nan_to_num_12(in_ptr0, in_ptr1, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp13 = tl.load(in_ptr1 + (0)).to(tl.float32)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    _tmp18 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2*x0) + (ks0*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = float("inf")
        tmp3 = tmp1 == tmp2
        tmp4 = float("-inf")
        tmp5 = tmp1 == tmp4
        tmp6 = libdevice.isnan(tmp0).to(tl.int1)
        tmp7 = 0.0
        tmp8 = tl.where(tmp6, tmp7, tmp0)
        tmp9 = -3.3895313892515355e+38
        tmp10 = tl.where(tmp5, tmp9, tmp8)
        tmp11 = 3.3895313892515355e+38
        tmp12 = tl.where(tmp3, tmp11, tmp10)
        tmp15 = tmp12 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = triton_helpers.maximum(_tmp18, tmp17)
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = triton_helpers.max2(_tmp18, 1)[:, None]
    tmp33 = tl.load(in_ptr1 + (0)).to(tl.float32)
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
    _tmp40 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr0 + (r1 + (2*x0) + (ks0*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = float("inf")
        tmp23 = tmp21 == tmp22
        tmp24 = float("-inf")
        tmp25 = tmp21 == tmp24
        tmp26 = libdevice.isnan(tmp20).to(tl.int1)
        tmp27 = 0.0
        tmp28 = tl.where(tmp26, tmp27, tmp20)
        tmp29 = -3.3895313892515355e+38
        tmp30 = tl.where(tmp25, tmp29, tmp28)
        tmp31 = 3.3895313892515355e+38
        tmp32 = tl.where(tmp23, tmp31, tmp30)
        tmp35 = tmp32 + tmp34
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tmp36 - tmp18
        tmp38 = tl.exp(tmp37)
        tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
        tmp41 = _tmp40 + tmp39
        _tmp40 = tl.where(rmask & xmask, tmp41, _tmp40)
    tmp40 = tl.sum(_tmp40, 1)[:, None]
    tmp55 = tl.load(in_ptr1 + (0)).to(tl.float32)
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp42 = tl.load(in_ptr0 + (r1 + (2*x0) + (ks0*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp43 = tmp42.to(tl.float32)
        tmp44 = float("inf")
        tmp45 = tmp43 == tmp44
        tmp46 = float("-inf")
        tmp47 = tmp43 == tmp46
        tmp48 = libdevice.isnan(tmp42).to(tl.int1)
        tmp49 = 0.0
        tmp50 = tl.where(tmp48, tmp49, tmp42)
        tmp51 = -3.3895313892515355e+38
        tmp52 = tl.where(tmp47, tmp51, tmp50)
        tmp53 = 3.3895313892515355e+38
        tmp54 = tl.where(tmp45, tmp53, tmp52)
        tmp57 = tmp54 + tmp56
        tmp58 = tmp57.to(tl.float32)
        tmp59 = tmp58 - tmp18
        tmp60 = tl.exp(tmp59)
        tmp61 = tmp60 / tmp40
        tmp62 = tmp61.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (2*x0) + (ks0*x0)), tmp62, rmask & xmask)
''')