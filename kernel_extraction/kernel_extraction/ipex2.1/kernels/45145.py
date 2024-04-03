

# Original file: ./eca_halonext26ts___60.0/eca_halonext26ts___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/kv/ckvcrvx7lyeigdas65i7ryqcymendkczwoprh6wm5atrh63pj5dt.py
# Source Nodes: [add_6, mul_5, softmax], Original ATen: [aten._softmax, aten.add, aten.mul]
# add_6 => add_50
# mul_5 => mul_90
# softmax => amax, convert_element_type_136, convert_element_type_137, div, exp, sub_22, sum_1
triton_red_fused__softmax_add_mul_27 = async_compile.triton('triton_red_fused__softmax_add_mul_27', '''
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
    size_hints=[262144, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_27', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_add_mul_27(in_ptr0, in_ptr1, in_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 262144
    rnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp25 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (144*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = 0.25
        tmp2 = tmp0 * tmp1
        tmp3 = 11 + (23*(x0 // 8)) + (r2 // 12)
        tmp4 = tl.full([1, 1], 192, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = (11 + (23*(x0 // 8)) + (r2 // 12)) % 24
        tmp7 = tl.full([1, 1], 23, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp5
        tmp10 = tl.load(in_ptr1 + ((23*((11 + (23*(x0 // 8)) + (r2 // 12)) // 24)) + (184*(x0 % 8)) + (1472*x1) + ((11 + (23*(x0 // 8)) + (r2 // 12)) % 24)), rmask & tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tl.where(tmp9, tmp10, 0.0)
        tmp12 = tl.where(tmp5, tmp11, 0.0)
        tmp13 = 11 + (23*(x0 % 8)) + (r2 % 12)
        tmp14 = tmp13 < tmp4
        tmp15 = (11 + (23*(x0 % 8)) + (r2 % 12)) % 24
        tmp16 = tmp15 < tmp7
        tmp17 = tmp16 & tmp14
        tmp18 = tl.load(in_ptr2 + ((23*(((11 + (23*(x0 % 8)) + (r2 % 12)) // 24) % 8)) + (184*(x0 // 8)) + (1472*x1) + ((11 + (23*(x0 % 8)) + (r2 % 12)) % 24)), rmask & tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tl.where(tmp17, tmp18, 0.0)
        tmp20 = tl.where(tmp14, tmp19, 0.0)
        tmp21 = tmp12 + tmp20
        tmp22 = tmp2 + tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = triton_helpers.maximum(_tmp25, tmp24)
        _tmp25 = tl.where(rmask, tmp26, _tmp25)
    tmp25 = triton_helpers.max2(_tmp25, 1)[:, None]
    _tmp54 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp27 = tl.load(in_ptr0 + (r2 + (144*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp28 = 0.25
        tmp29 = tmp27 * tmp28
        tmp30 = 11 + (23*(x0 // 8)) + (r2 // 12)
        tmp31 = tl.full([1, 1], 192, tl.int64)
        tmp32 = tmp30 < tmp31
        tmp33 = (11 + (23*(x0 // 8)) + (r2 // 12)) % 24
        tmp34 = tl.full([1, 1], 23, tl.int64)
        tmp35 = tmp33 < tmp34
        tmp36 = tmp35 & tmp32
        tmp37 = tl.load(in_ptr1 + ((23*((11 + (23*(x0 // 8)) + (r2 // 12)) // 24)) + (184*(x0 % 8)) + (1472*x1) + ((11 + (23*(x0 // 8)) + (r2 // 12)) % 24)), rmask & tmp36, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp38 = tl.where(tmp36, tmp37, 0.0)
        tmp39 = tl.where(tmp32, tmp38, 0.0)
        tmp40 = 11 + (23*(x0 % 8)) + (r2 % 12)
        tmp41 = tmp40 < tmp31
        tmp42 = (11 + (23*(x0 % 8)) + (r2 % 12)) % 24
        tmp43 = tmp42 < tmp34
        tmp44 = tmp43 & tmp41
        tmp45 = tl.load(in_ptr2 + ((23*(((11 + (23*(x0 % 8)) + (r2 % 12)) // 24) % 8)) + (184*(x0 // 8)) + (1472*x1) + ((11 + (23*(x0 % 8)) + (r2 % 12)) % 24)), rmask & tmp44, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp46 = tl.where(tmp44, tmp45, 0.0)
        tmp47 = tl.where(tmp41, tmp46, 0.0)
        tmp48 = tmp39 + tmp47
        tmp49 = tmp29 + tmp48
        tmp50 = tmp49.to(tl.float32)
        tmp51 = tmp50 - tmp25
        tmp52 = tl.exp(tmp51)
        tmp53 = tl.broadcast_to(tmp52, [XBLOCK, RBLOCK])
        tmp55 = _tmp54 + tmp53
        _tmp54 = tl.where(rmask, tmp55, _tmp54)
    tmp54 = tl.sum(_tmp54, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp56 = tl.load(in_ptr0 + (r2 + (144*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp57 = 0.25
        tmp58 = tmp56 * tmp57
        tmp59 = 11 + (23*(x0 // 8)) + (r2 // 12)
        tmp60 = tl.full([1, 1], 192, tl.int64)
        tmp61 = tmp59 < tmp60
        tmp62 = (11 + (23*(x0 // 8)) + (r2 // 12)) % 24
        tmp63 = tl.full([1, 1], 23, tl.int64)
        tmp64 = tmp62 < tmp63
        tmp65 = tmp64 & tmp61
        tmp66 = tl.load(in_ptr1 + ((23*((11 + (23*(x0 // 8)) + (r2 // 12)) // 24)) + (184*(x0 % 8)) + (1472*x1) + ((11 + (23*(x0 // 8)) + (r2 // 12)) % 24)), rmask & tmp65, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp67 = tl.where(tmp65, tmp66, 0.0)
        tmp68 = tl.where(tmp61, tmp67, 0.0)
        tmp69 = 11 + (23*(x0 % 8)) + (r2 % 12)
        tmp70 = tmp69 < tmp60
        tmp71 = (11 + (23*(x0 % 8)) + (r2 % 12)) % 24
        tmp72 = tmp71 < tmp63
        tmp73 = tmp72 & tmp70
        tmp74 = tl.load(in_ptr2 + ((23*(((11 + (23*(x0 % 8)) + (r2 % 12)) // 24) % 8)) + (184*(x0 // 8)) + (1472*x1) + ((11 + (23*(x0 % 8)) + (r2 % 12)) % 24)), rmask & tmp73, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp75 = tl.where(tmp73, tmp74, 0.0)
        tmp76 = tl.where(tmp70, tmp75, 0.0)
        tmp77 = tmp68 + tmp76
        tmp78 = tmp58 + tmp77
        tmp79 = tmp78.to(tl.float32)
        tmp80 = tmp79 - tmp25
        tmp81 = tl.exp(tmp80)
        tmp82 = tmp81 / tmp54
        tmp83 = tmp82.to(tl.float32)
        tl.store(out_ptr3 + (r2 + (144*x3)), tmp83, rmask)
''')
