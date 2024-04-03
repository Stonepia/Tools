

# Original file: ./cm3leon_generate__30_inference_70.10/cm3leon_generate__30_inference_70.10_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/tz/ctzqpkcpdkwjjcdmu356mowzo2n3jque7nwpukzadswmnruhyxda.py
# Source Nodes: [iadd_19, nan_to_num, nan_to_num_18, softmax_18, triu], Original ATen: [aten._softmax, aten.add, aten.nan_to_num, aten.triu]
# iadd_19 => add_130
# nan_to_num => full_default_3, full_default_4
# nan_to_num_18 => eq_74, eq_75, isnan_18, where_55, where_56, where_57
# softmax_18 => amax_18, div_18, exp_18, sub_56, sum_19
# triu => full_default_1
triton_red_fused__softmax_add_nan_to_num_triu_66 = async_compile.triton('triton_red_fused__softmax_add_nan_to_num_triu_66', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_nan_to_num_triu_66', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_add_nan_to_num_triu_66(in_ptr0, in_ptr1, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp12 = tl.load(in_ptr1 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    _tmp16 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (19*x0) + (ks0*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = float("inf")
        tmp2 = tmp0 == tmp1
        tmp3 = float("-inf")
        tmp4 = tmp0 == tmp3
        tmp5 = libdevice.isnan(tmp0).to(tl.int1)
        tmp6 = 0.0
        tmp7 = tl.where(tmp5, tmp6, tmp0)
        tmp8 = -3.4028234663852886e+38
        tmp9 = tl.where(tmp4, tmp8, tmp7)
        tmp10 = 3.4028234663852886e+38
        tmp11 = tl.where(tmp2, tmp10, tmp9)
        tmp14 = tmp11 + tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = triton_helpers.maximum(_tmp16, tmp15)
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp16 = triton_helpers.max2(_tmp16, 1)[:, None]
    tmp30 = tl.load(in_ptr1 + (0))
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    _tmp36 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr0 + (r1 + (19*x0) + (ks0*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = float("inf")
        tmp20 = tmp18 == tmp19
        tmp21 = float("-inf")
        tmp22 = tmp18 == tmp21
        tmp23 = libdevice.isnan(tmp18).to(tl.int1)
        tmp24 = 0.0
        tmp25 = tl.where(tmp23, tmp24, tmp18)
        tmp26 = -3.4028234663852886e+38
        tmp27 = tl.where(tmp22, tmp26, tmp25)
        tmp28 = 3.4028234663852886e+38
        tmp29 = tl.where(tmp20, tmp28, tmp27)
        tmp32 = tmp29 + tmp31
        tmp33 = tmp32 - tmp16
        tmp34 = tl.exp(tmp33)
        tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
        tmp37 = _tmp36 + tmp35
        _tmp36 = tl.where(rmask & xmask, tmp37, _tmp36)
    tmp36 = tl.sum(_tmp36, 1)[:, None]
    tmp50 = tl.load(in_ptr1 + (0))
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp38 = tl.load(in_ptr0 + (r1 + (19*x0) + (ks0*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp39 = float("inf")
        tmp40 = tmp38 == tmp39
        tmp41 = float("-inf")
        tmp42 = tmp38 == tmp41
        tmp43 = libdevice.isnan(tmp38).to(tl.int1)
        tmp44 = 0.0
        tmp45 = tl.where(tmp43, tmp44, tmp38)
        tmp46 = -3.4028234663852886e+38
        tmp47 = tl.where(tmp42, tmp46, tmp45)
        tmp48 = 3.4028234663852886e+38
        tmp49 = tl.where(tmp40, tmp48, tmp47)
        tmp52 = tmp49 + tmp51
        tmp53 = tmp52 - tmp16
        tmp54 = tl.exp(tmp53)
        tmp55 = tmp54 / tmp36
        tl.store(out_ptr2 + (r1 + (19*x0) + (ks0*x0)), tmp55, rmask & xmask)
''')
