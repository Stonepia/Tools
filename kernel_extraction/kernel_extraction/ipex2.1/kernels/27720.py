

# Original file: ./hf_T5_generate___60.0/hf_T5_generate___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/jq/cjq6qpl7gleu4xzledqrnh4cttp6roy53ukxrnbpssr3kocwfpt4.py
# Source Nodes: [softmax], Original ATen: [aten._softmax]
# softmax => amax, div_2, exp, sub_2, sum_1
triton_red_fused__softmax_2 = async_compile.triton('triton_red_fused__softmax_2', '''
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
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_2(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp35 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp25 = tl.load(in_ptr1 + (x1 + (8*tmp24)), rmask, eviction_policy='evict_last')
        tmp27 = tmp26.to(tl.float32)
        tmp28 = 1.0
        tmp29 = tmp28 - tmp27
        tmp30 = -3.4028234663852886e+38
        tmp31 = tmp29 * tmp30
        tmp32 = tmp25 + tmp31
        tmp33 = tmp0 + tmp32
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
        tmp36 = triton_helpers.maximum(_tmp35, tmp34)
        _tmp35 = tl.where(rmask, tmp36, _tmp35)
    tmp35 = triton_helpers.max2(_tmp35, 1)[:, None]
    _tmp74 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp37 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp63 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp38 = r2 + ((-1)*x0)
        tmp39 = tl.full([1, 1], 0, tl.int64)
        tmp40 = tmp38 > tmp39
        tmp41 = tmp40.to(tl.int64)
        tmp42 = tl.full([1, 1], 16, tl.int64)
        tmp43 = tmp41 * tmp42
        tmp44 = tmp43 + tmp39
        tmp45 = tl.abs(tmp38)
        tmp46 = tl.full([1, 1], 8, tl.int64)
        tmp47 = tmp45 < tmp46
        tmp48 = tmp45.to(tl.float32)
        tmp49 = 8.0
        tmp50 = tmp48 / tmp49
        tmp51 = tl.log(tmp50)
        tmp52 = 2.772588722239781
        tmp53 = tmp51 / tmp52
        tmp54 = tmp53 * tmp49
        tmp55 = tmp54.to(tl.int64)
        tmp56 = tmp55 + tmp46
        tmp57 = tl.full([1, 1], 15, tl.int64)
        tmp58 = triton_helpers.minimum(tmp56, tmp57)
        tmp59 = tl.where(tmp47, tmp45, tmp58)
        tmp60 = tmp44 + tmp59
        tmp61 = tl.where(tmp60 < 0, tmp60 + 32, tmp60)
        # tl.device_assert(((0 <= tmp61) & (tmp61 < 32)) | ~rmask, "index out of bounds: 0 <= tmp61 < 32")
        tmp62 = tl.load(in_ptr1 + (x1 + (8*tmp61)), rmask, eviction_policy='evict_last')
        tmp64 = tmp63.to(tl.float32)
        tmp65 = 1.0
        tmp66 = tmp65 - tmp64
        tmp67 = -3.4028234663852886e+38
        tmp68 = tmp66 * tmp67
        tmp69 = tmp62 + tmp68
        tmp70 = tmp37 + tmp69
        tmp71 = tmp70 - tmp35
        tmp72 = tl.exp(tmp71)
        tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
        tmp75 = _tmp74 + tmp73
        _tmp74 = tl.where(rmask, tmp75, _tmp74)
    tmp74 = tl.sum(_tmp74, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp76 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp102 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp77 = r2 + ((-1)*x0)
        tmp78 = tl.full([1, 1], 0, tl.int64)
        tmp79 = tmp77 > tmp78
        tmp80 = tmp79.to(tl.int64)
        tmp81 = tl.full([1, 1], 16, tl.int64)
        tmp82 = tmp80 * tmp81
        tmp83 = tmp82 + tmp78
        tmp84 = tl.abs(tmp77)
        tmp85 = tl.full([1, 1], 8, tl.int64)
        tmp86 = tmp84 < tmp85
        tmp87 = tmp84.to(tl.float32)
        tmp88 = 8.0
        tmp89 = tmp87 / tmp88
        tmp90 = tl.log(tmp89)
        tmp91 = 2.772588722239781
        tmp92 = tmp90 / tmp91
        tmp93 = tmp92 * tmp88
        tmp94 = tmp93.to(tl.int64)
        tmp95 = tmp94 + tmp85
        tmp96 = tl.full([1, 1], 15, tl.int64)
        tmp97 = triton_helpers.minimum(tmp95, tmp96)
        tmp98 = tl.where(tmp86, tmp84, tmp97)
        tmp99 = tmp83 + tmp98
        tmp100 = tl.where(tmp99 < 0, tmp99 + 32, tmp99)
        # tl.device_assert(((0 <= tmp100) & (tmp100 < 32)) | ~rmask, "index out of bounds: 0 <= tmp100 < 32")
        tmp101 = tl.load(in_ptr1 + (x1 + (8*tmp100)), rmask, eviction_policy='evict_first')
        tmp103 = tmp102.to(tl.float32)
        tmp104 = 1.0
        tmp105 = tmp104 - tmp103
        tmp106 = -3.4028234663852886e+38
        tmp107 = tmp105 * tmp106
        tmp108 = tmp101 + tmp107
        tmp109 = tmp76 + tmp108
        tmp110 = tmp109 - tmp35
        tmp111 = tl.exp(tmp110)
        tmp112 = tmp111 / tmp74
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp112, rmask)
''')
