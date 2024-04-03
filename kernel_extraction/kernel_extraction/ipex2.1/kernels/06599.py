

# Original file: ./hf_T5_base___60.0/hf_T5_base___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/gw/cgwjrytsq53ogox4tcnw55tuaqcg6suh5qrkl64vhg7eofdptutk.py
# Source Nodes: [float_2, softmax, type_as], Original ATen: [aten._softmax, aten._to_copy]
# float_2 => convert_element_type_6
# softmax => amax, div_2, exp, sub_2, sum_1
# type_as => convert_element_type_7
triton_red_fused__softmax__to_copy_7 = async_compile.triton('triton_red_fused__softmax__to_copy_7', '''
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
    size_hints=[32768, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_7(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp29 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp25 = tl.load(in_ptr1 + (x1 + (12*tmp24)), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp26 = tmp0 + tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = triton_helpers.maximum(_tmp29, tmp28)
        _tmp29 = tl.where(rmask, tmp30, _tmp29)
    tmp29 = triton_helpers.max2(_tmp29, 1)[:, None]
    _tmp62 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp31 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp32 = r2 + ((-1)*x0)
        tmp33 = tl.full([1, 1], 0, tl.int64)
        tmp34 = tmp32 > tmp33
        tmp35 = tmp34.to(tl.int64)
        tmp36 = tl.full([1, 1], 16, tl.int64)
        tmp37 = tmp35 * tmp36
        tmp38 = tmp37 + tmp33
        tmp39 = tl.abs(tmp32)
        tmp40 = tl.full([1, 1], 8, tl.int64)
        tmp41 = tmp39 < tmp40
        tmp42 = tmp39.to(tl.float32)
        tmp43 = 8.0
        tmp44 = tmp42 / tmp43
        tmp45 = tl.log(tmp44)
        tmp46 = 2.772588722239781
        tmp47 = tmp45 / tmp46
        tmp48 = tmp47 * tmp43
        tmp49 = tmp48.to(tl.int64)
        tmp50 = tmp49 + tmp40
        tmp51 = tl.full([1, 1], 15, tl.int64)
        tmp52 = triton_helpers.minimum(tmp50, tmp51)
        tmp53 = tl.where(tmp41, tmp39, tmp52)
        tmp54 = tmp38 + tmp53
        tmp55 = tl.where(tmp54 < 0, tmp54 + 32, tmp54)
        # tl.device_assert(((0 <= tmp55) & (tmp55 < 32)) | ~rmask, "index out of bounds: 0 <= tmp55 < 32")
        tmp56 = tl.load(in_ptr1 + (x1 + (12*tmp55)), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp57 = tmp31 + tmp56
        tmp58 = tmp57.to(tl.float32)
        tmp59 = tmp58 - tmp29
        tmp60 = tl.exp(tmp59)
        tmp61 = tl.broadcast_to(tmp60, [XBLOCK, RBLOCK])
        tmp63 = _tmp62 + tmp61
        _tmp62 = tl.where(rmask, tmp63, _tmp62)
    tmp62 = tl.sum(_tmp62, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp64 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp65 = r2 + ((-1)*x0)
        tmp66 = tl.full([1, 1], 0, tl.int64)
        tmp67 = tmp65 > tmp66
        tmp68 = tmp67.to(tl.int64)
        tmp69 = tl.full([1, 1], 16, tl.int64)
        tmp70 = tmp68 * tmp69
        tmp71 = tmp70 + tmp66
        tmp72 = tl.abs(tmp65)
        tmp73 = tl.full([1, 1], 8, tl.int64)
        tmp74 = tmp72 < tmp73
        tmp75 = tmp72.to(tl.float32)
        tmp76 = 8.0
        tmp77 = tmp75 / tmp76
        tmp78 = tl.log(tmp77)
        tmp79 = 2.772588722239781
        tmp80 = tmp78 / tmp79
        tmp81 = tmp80 * tmp76
        tmp82 = tmp81.to(tl.int64)
        tmp83 = tmp82 + tmp73
        tmp84 = tl.full([1, 1], 15, tl.int64)
        tmp85 = triton_helpers.minimum(tmp83, tmp84)
        tmp86 = tl.where(tmp74, tmp72, tmp85)
        tmp87 = tmp71 + tmp86
        tmp88 = tl.where(tmp87 < 0, tmp87 + 32, tmp87)
        # tl.device_assert(((0 <= tmp88) & (tmp88 < 32)) | ~rmask, "index out of bounds: 0 <= tmp88 < 32")
        tmp89 = tl.load(in_ptr1 + (x1 + (12*tmp88)), rmask, eviction_policy='evict_first').to(tl.float32)
        tmp90 = tmp64 + tmp89
        tmp91 = tmp90.to(tl.float32)
        tmp92 = tmp91 - tmp29
        tmp93 = tl.exp(tmp92)
        tmp94 = tmp93 / tmp62
        tmp95 = tmp94.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp95, rmask)
''')
