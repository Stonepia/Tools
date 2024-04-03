

# Original file: ./cm3leon_generate__30_inference_70.10/cm3leon_generate__30_inference_70.10_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/yz/cyzn4hbi5ji44ztalipbh7m7mu5ukmumap6q2mq6uov7x2dsnki5.py
# Source Nodes: [iadd_15, nan_to_num, nan_to_num_14, softmax_14, type_as_16], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.nan_to_num]
# iadd_15 => add_102
# nan_to_num => full_default_2, full_default_3, full_default_4
# nan_to_num_14 => convert_element_type_306, eq_58, eq_59, isnan_14, where_43, where_44, where_45
# softmax_14 => amax_14, div_14, exp_14, sub_44, sum_15
# type_as_16 => convert_element_type_310
triton_red_fused__softmax__to_copy_add_nan_to_num_54 = async_compile.triton('triton_red_fused__softmax__to_copy_add_nan_to_num_54', '''
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
    size_hints=[16, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: 'i32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_add_nan_to_num_54', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_add_nan_to_num_54(in_ptr0, in_ptr1, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp14 = tl.load(in_ptr1 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    _tmp18 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (15*x0) + (ks0*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = float("inf")
        tmp3 = tmp1 == tmp2
        tmp4 = float("-inf")
        tmp5 = tmp1 == tmp4
        tmp6 = libdevice.isnan(tmp0).to(tl.int1)
        tmp7 = 0.0
        tmp8 = tl.where(tmp6, tmp7, tmp0)
        tmp9 = -65504.0
        tmp10 = tl.where(tmp5, tmp9, tmp8)
        tmp11 = 65504.0
        tmp12 = tl.where(tmp3, tmp11, tmp10)
        tmp13 = tmp12.to(tl.float32)
        tmp16 = tmp13 + tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = triton_helpers.maximum(_tmp18, tmp17)
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = triton_helpers.max2(_tmp18, 1)[:, None]
    tmp34 = tl.load(in_ptr1 + (0))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
    _tmp40 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr0 + (r1 + (15*x0) + (ks0*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = float("inf")
        tmp23 = tmp21 == tmp22
        tmp24 = float("-inf")
        tmp25 = tmp21 == tmp24
        tmp26 = libdevice.isnan(tmp20).to(tl.int1)
        tmp27 = 0.0
        tmp28 = tl.where(tmp26, tmp27, tmp20)
        tmp29 = -65504.0
        tmp30 = tl.where(tmp25, tmp29, tmp28)
        tmp31 = 65504.0
        tmp32 = tl.where(tmp23, tmp31, tmp30)
        tmp33 = tmp32.to(tl.float32)
        tmp36 = tmp33 + tmp35
        tmp37 = tmp36 - tmp18
        tmp38 = tl.exp(tmp37)
        tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
        tmp41 = _tmp40 + tmp39
        _tmp40 = tl.where(rmask & xmask, tmp41, _tmp40)
    tmp40 = tl.sum(_tmp40, 1)[:, None]
    tmp56 = tl.load(in_ptr1 + (0))
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp42 = tl.load(in_ptr0 + (r1 + (15*x0) + (ks0*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp43 = tmp42.to(tl.float32)
        tmp44 = float("inf")
        tmp45 = tmp43 == tmp44
        tmp46 = float("-inf")
        tmp47 = tmp43 == tmp46
        tmp48 = libdevice.isnan(tmp42).to(tl.int1)
        tmp49 = 0.0
        tmp50 = tl.where(tmp48, tmp49, tmp42)
        tmp51 = -65504.0
        tmp52 = tl.where(tmp47, tmp51, tmp50)
        tmp53 = 65504.0
        tmp54 = tl.where(tmp45, tmp53, tmp52)
        tmp55 = tmp54.to(tl.float32)
        tmp58 = tmp55 + tmp57
        tmp59 = tmp58 - tmp18
        tmp60 = tl.exp(tmp59)
        tmp61 = tmp60 / tmp40
        tmp62 = tmp61.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (15*x0) + (ks0*x0)), tmp62, rmask & xmask)
''')
