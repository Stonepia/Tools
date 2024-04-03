

# Original file: ./hf_T5_large___60.0/hf_T5_large___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ze/czes4gsat5fd6yawqq46jrjcqxjn2uqfrpzjm6lrqvg7bw75s3dk.py
# Source Nodes: [softmax], Original ATen: [aten._softmax]
# softmax => amax, div_2, exp, sub_2, sum_1
triton_red_fused__softmax_5 = async_compile.triton('triton_red_fused__softmax_5', '''
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
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_5(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp28 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = r2 + ((-1)*x0)
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 > tmp2
        tmp4 = tmp3.to(tl.int64)
        tmp5 = tl.full([1, 1], 16, tl.int64)
        tmp6 = tmp4 * tmp5
        tmp7 = tmp6 + tmp2
        tmp8 = tl.abs(tmp1)
        tmp9 = tl.full([1, 1], 8, tl.int64)
        tmp10 = tmp8 < tmp9
        tmp11 = tmp8.to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = tl.log(tmp13)
        tmp15 = 2.772588722239781
        tmp16 = tmp14 / tmp15
        tmp17 = tmp16 * tmp12
        tmp18 = tmp17.to(tl.int64)
        tmp19 = tmp18 + tmp9
        tmp20 = tl.full([1, 1], 15, tl.int64)
        tmp21 = triton_helpers.minimum(tmp19, tmp20)
        tmp22 = tl.where(tmp10, tmp8, tmp21)
        tmp23 = tmp7 + tmp22
        tmp24 = tl.where(tmp23 < 0, tmp23 + 32, tmp23)
        # tl.device_assert(((0 <= tmp24) & (tmp24 < 32)) | ~rmask, "index out of bounds: 0 <= tmp24 < 32")
        tmp25 = tl.load(in_ptr1 + (x1 + (16*tmp24)), rmask, eviction_policy='evict_last')
        tmp26 = tmp0 + tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = triton_helpers.maximum(_tmp28, tmp27)
        _tmp28 = tl.where(rmask, tmp29, _tmp28)
    tmp28 = triton_helpers.max2(_tmp28, 1)[:, None]
    _tmp60 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp30 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = r2 + ((-1)*x0)
        tmp32 = tl.full([1, 1], 0, tl.int64)
        tmp33 = tmp31 > tmp32
        tmp34 = tmp33.to(tl.int64)
        tmp35 = tl.full([1, 1], 16, tl.int64)
        tmp36 = tmp34 * tmp35
        tmp37 = tmp36 + tmp32
        tmp38 = tl.abs(tmp31)
        tmp39 = tl.full([1, 1], 8, tl.int64)
        tmp40 = tmp38 < tmp39
        tmp41 = tmp38.to(tl.float32)
        tmp42 = 8.0
        tmp43 = tmp41 / tmp42
        tmp44 = tl.log(tmp43)
        tmp45 = 2.772588722239781
        tmp46 = tmp44 / tmp45
        tmp47 = tmp46 * tmp42
        tmp48 = tmp47.to(tl.int64)
        tmp49 = tmp48 + tmp39
        tmp50 = tl.full([1, 1], 15, tl.int64)
        tmp51 = triton_helpers.minimum(tmp49, tmp50)
        tmp52 = tl.where(tmp40, tmp38, tmp51)
        tmp53 = tmp37 + tmp52
        tmp54 = tl.where(tmp53 < 0, tmp53 + 32, tmp53)
        # tl.device_assert(((0 <= tmp54) & (tmp54 < 32)) | ~rmask, "index out of bounds: 0 <= tmp54 < 32")
        tmp55 = tl.load(in_ptr1 + (x1 + (16*tmp54)), rmask, eviction_policy='evict_last')
        tmp56 = tmp30 + tmp55
        tmp57 = tmp56 - tmp28
        tmp58 = tl.exp(tmp57)
        tmp59 = tl.broadcast_to(tmp58, [XBLOCK, RBLOCK])
        tmp61 = _tmp60 + tmp59
        _tmp60 = tl.where(rmask, tmp61, _tmp60)
    tmp60 = tl.sum(_tmp60, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp62 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp63 = r2 + ((-1)*x0)
        tmp64 = tl.full([1, 1], 0, tl.int64)
        tmp65 = tmp63 > tmp64
        tmp66 = tmp65.to(tl.int64)
        tmp67 = tl.full([1, 1], 16, tl.int64)
        tmp68 = tmp66 * tmp67
        tmp69 = tmp68 + tmp64
        tmp70 = tl.abs(tmp63)
        tmp71 = tl.full([1, 1], 8, tl.int64)
        tmp72 = tmp70 < tmp71
        tmp73 = tmp70.to(tl.float32)
        tmp74 = 8.0
        tmp75 = tmp73 / tmp74
        tmp76 = tl.log(tmp75)
        tmp77 = 2.772588722239781
        tmp78 = tmp76 / tmp77
        tmp79 = tmp78 * tmp74
        tmp80 = tmp79.to(tl.int64)
        tmp81 = tmp80 + tmp71
        tmp82 = tl.full([1, 1], 15, tl.int64)
        tmp83 = triton_helpers.minimum(tmp81, tmp82)
        tmp84 = tl.where(tmp72, tmp70, tmp83)
        tmp85 = tmp69 + tmp84
        tmp86 = tl.where(tmp85 < 0, tmp85 + 32, tmp85)
        # tl.device_assert(((0 <= tmp86) & (tmp86 < 32)) | ~rmask, "index out of bounds: 0 <= tmp86 < 32")
        tmp87 = tl.load(in_ptr1 + (x1 + (16*tmp86)), rmask, eviction_policy='evict_first')
        tmp88 = tmp62 + tmp87
        tmp89 = tmp88 - tmp28
        tmp90 = tl.exp(tmp89)
        tmp91 = tmp90 / tmp60
        tl.store(out_ptr2 + (r2 + (512*x3)), tmp91, rmask)
''')
