

# Original file: ./hf_T5_generate___60.0/hf_T5_generate___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/6p/c6pcavcnaykbkhelveupngrxxo3nzsyg5snjn2elkhisf74vfbv6.py
# Source Nodes: [float_2, softmax, type_as], Original ATen: [aten._softmax, aten._to_copy]
# float_2 => convert_element_type_6
# softmax => amax, div_2, exp, sub_2, sum_1
# type_as => convert_element_type_7
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
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*i64', 3: '*fp32', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_2(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp36 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp25 = tl.load(in_ptr1 + (x1 + (8*tmp24)), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tmp26.to(tl.float32)
        tmp28 = 1.0
        tmp29 = tmp28 - tmp27
        tmp30 = -3.3895313892515355e+38
        tmp31 = tmp29 * tmp30
        tmp32 = tmp25 + tmp31
        tmp33 = tmp0 + tmp32
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
        tmp37 = triton_helpers.maximum(_tmp36, tmp35)
        _tmp36 = tl.where(rmask, tmp37, _tmp36)
    tmp36 = triton_helpers.max2(_tmp36, 1)[:, None]
    _tmp76 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp38 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp64 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp39 = r2 + ((-1)*x0)
        tmp40 = tl.full([1, 1], 0, tl.int64)
        tmp41 = tmp39 > tmp40
        tmp42 = tmp41.to(tl.int64)
        tmp43 = tl.full([1, 1], 16, tl.int64)
        tmp44 = tmp42 * tmp43
        tmp45 = tmp44 + tmp40
        tmp46 = tl.abs(tmp39)
        tmp47 = tl.full([1, 1], 8, tl.int64)
        tmp48 = tmp46 < tmp47
        tmp49 = tmp46.to(tl.float32)
        tmp50 = 8.0
        tmp51 = tmp49 / tmp50
        tmp52 = tl.log(tmp51)
        tmp53 = 2.772588722239781
        tmp54 = tmp52 / tmp53
        tmp55 = tmp54 * tmp50
        tmp56 = tmp55.to(tl.int64)
        tmp57 = tmp56 + tmp47
        tmp58 = tl.full([1, 1], 15, tl.int64)
        tmp59 = triton_helpers.minimum(tmp57, tmp58)
        tmp60 = tl.where(tmp48, tmp46, tmp59)
        tmp61 = tmp45 + tmp60
        tmp62 = tl.where(tmp61 < 0, tmp61 + 32, tmp61)
        # tl.device_assert(((0 <= tmp62) & (tmp62 < 32)) | ~rmask, "index out of bounds: 0 <= tmp62 < 32")
        tmp63 = tl.load(in_ptr1 + (x1 + (8*tmp62)), rmask, eviction_policy='evict_first').to(tl.float32)
        tmp65 = tmp64.to(tl.float32)
        tmp66 = 1.0
        tmp67 = tmp66 - tmp65
        tmp68 = -3.3895313892515355e+38
        tmp69 = tmp67 * tmp68
        tmp70 = tmp63 + tmp69
        tmp71 = tmp38 + tmp70
        tmp72 = tmp71.to(tl.float32)
        tmp73 = tmp72 - tmp36
        tmp74 = tl.exp(tmp73)
        tmp75 = tl.broadcast_to(tmp74, [XBLOCK, RBLOCK])
        tmp77 = _tmp76 + tmp75
        _tmp76 = tl.where(rmask, tmp77, _tmp76)
        tl.store(out_ptr1 + (r2 + (2048*x3)), tmp73, rmask)
    tmp76 = tl.sum(_tmp76, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp78 = tl.load(out_ptr1 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp79 = tl.exp(tmp78)
        tmp80 = tmp79 / tmp76
        tmp81 = tmp80.to(tl.float32)
        tl.store(out_ptr3 + (r2 + (2048*x3)), tmp81, rmask)
''')
