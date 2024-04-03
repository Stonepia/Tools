

# Original file: ./cm3leon_generate__28_inference_68.8/cm3leon_generate__28_inference_68.8_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/md/cmdjqp3asvis3ukj3ixeikd2hnsc654rzi2b4ib2wigex2nrkfms.py
# Source Nodes: [iadd_24, nan_to_num, nan_to_num_23, softmax_23, type_as_25], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.nan_to_num]
# iadd_24 => add_165
# nan_to_num => full_default_2, full_default_3, full_default_4
# nan_to_num_23 => convert_element_type_190, eq_94, eq_95, isnan_23, where_70, where_71, where_72
# softmax_23 => amax_23, convert_element_type_192, div_23, exp_23, sub_71, sum_24
# type_as_25 => convert_element_type_193
triton_red_fused__softmax__to_copy_add_nan_to_num_81 = async_compile.triton('triton_red_fused__softmax__to_copy_add_nan_to_num_81', '''
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
    size_hints=[16, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_add_nan_to_num_81', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_add_nan_to_num_81(in_ptr0, in_ptr1, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (r1 + (24*x0) + (ks0*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp20 = tl.load(in_ptr0 + (r1 + (24*x0) + (ks0*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp42 = tl.load(in_ptr0 + (r1 + (24*x0) + (ks0*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
        tmp57 = tmp54 + tmp56
        tmp58 = tmp57.to(tl.float32)
        tmp59 = tmp58 - tmp18
        tmp60 = tl.exp(tmp59)
        tmp61 = tmp60 / tmp40
        tmp62 = tmp61.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (24*x0) + (ks0*x0)), tmp62, rmask & xmask)
''')
