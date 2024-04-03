

# Original file: ./hf_T5_generate___60.0/hf_T5_generate___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/hi/chipbtotn6hfsudd3tmtlav64h7wz2zydi52k55ex4ss7zngvsx6.py
# Source Nodes: [add_3, float_7, iadd_6, mul, softmax_5, sub, to, type_as_5], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.mul, aten.rsub]
# add_3 => add_4
# float_7 => convert_element_type_86
# iadd_6 => add_30, convert_element_type_85
# mul => mul
# softmax_5 => amax_5, div_7, exp_5, sub_7, sum_6
# sub => sub
# to => convert_element_type
# type_as_5 => convert_element_type_87
triton_red_fused__softmax__to_copy_add_mul_rsub_12 = async_compile.triton('triton_red_fused__softmax__to_copy_add_mul_rsub_12', '''
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
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*i64', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_add_mul_rsub_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_add_mul_rsub_12(in_out_ptr0, in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp26 = tl.load(in_ptr0 + (x1 + (8*tmp25)), rmask, eviction_policy='evict_first')
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
        tl.store(in_out_ptr0 + (r2 + (2048*x3)), tmp35, rmask)
    tmp38 = triton_helpers.max2(_tmp38, 1)[:, None]
    _tmp45 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp40 = tl.load(in_out_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tmp41 - tmp38
        tmp43 = tl.exp(tmp42)
        tmp44 = tl.broadcast_to(tmp43, [XBLOCK, RBLOCK])
        tmp46 = _tmp45 + tmp44
        _tmp45 = tl.where(rmask, tmp46, _tmp45)
    tmp45 = tl.sum(_tmp45, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp47 = tl.load(in_out_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp48 = tmp47.to(tl.float32)
        tmp49 = tmp48 - tmp38
        tmp50 = tl.exp(tmp49)
        tmp51 = tmp50 / tmp45
        tmp52 = tmp51.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp52, rmask)
''')
