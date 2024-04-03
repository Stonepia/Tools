

# Original file: ./MT5ForConditionalGeneration__0_forward_205.0/MT5ForConditionalGeneration__0_forward_205.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/q7/cq7y27b3h6q2xiqymsptpsahkwiuzdgvsinhnnqfglh2pazoi3g7.py
# Source Nodes: [dropout, float_2, softmax, type_as], Original ATen: [aten._softmax, aten._to_copy, aten.native_dropout]
# dropout => gt_2, mul_7, mul_8
# float_2 => convert_element_type_10
# softmax => amax, div_2, exp, sub_2, sum_1
# type_as => convert_element_type_11
triton_red_fused__softmax__to_copy_native_dropout_5 = async_compile.triton('triton_red_fused__softmax__to_copy_native_dropout_5', '''
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp16', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_native_dropout_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_native_dropout_5(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr3, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128) % 6
    _tmp33 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = r3 + ((-1)*x0)
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
        tmp26 = tl.load(in_ptr1 + (x1 + (6*tmp25)), rmask, eviction_policy='evict_last')
        tmp27 = 0.0
        tmp28 = tmp26 + tmp27
        tmp29 = tmp1 + tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp34 = triton_helpers.maximum(_tmp33, tmp32)
        _tmp33 = tl.where(rmask, tmp34, _tmp33)
    tmp33 = triton_helpers.max2(_tmp33, 1)[:, None]
    _tmp70 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp35 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp36 = tmp35.to(tl.float32)
        tmp37 = r3 + ((-1)*x0)
        tmp38 = tl.full([1, 1], 0, tl.int64)
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.int64)
        tmp41 = tl.full([1, 1], 16, tl.int64)
        tmp42 = tmp40 * tmp41
        tmp43 = tmp42 + tmp38
        tmp44 = tl.abs(tmp37)
        tmp45 = tl.full([1, 1], 8, tl.int64)
        tmp46 = tmp44 < tmp45
        tmp47 = tmp44.to(tl.float32)
        tmp48 = 8.0
        tmp49 = tmp47 / tmp48
        tmp50 = tl.log(tmp49)
        tmp51 = 2.772588722239781
        tmp52 = tmp50 / tmp51
        tmp53 = tmp52 * tmp48
        tmp54 = tmp53.to(tl.int64)
        tmp55 = tmp54 + tmp45
        tmp56 = tl.full([1, 1], 15, tl.int64)
        tmp57 = triton_helpers.minimum(tmp55, tmp56)
        tmp58 = tl.where(tmp46, tmp44, tmp57)
        tmp59 = tmp43 + tmp58
        tmp60 = tl.where(tmp59 < 0, tmp59 + 32, tmp59)
        # tl.device_assert(((0 <= tmp60) & (tmp60 < 32)) | ~rmask, "index out of bounds: 0 <= tmp60 < 32")
        tmp61 = tl.load(in_ptr1 + (x1 + (6*tmp60)), rmask, eviction_policy='evict_first')
        tmp62 = 0.0
        tmp63 = tmp61 + tmp62
        tmp64 = tmp36 + tmp63
        tmp65 = tmp64.to(tl.float32)
        tmp66 = tmp65.to(tl.float32)
        tmp67 = tmp66 - tmp33
        tmp68 = tl.exp(tmp67)
        tmp69 = tl.broadcast_to(tmp68, [XBLOCK, RBLOCK])
        tmp71 = _tmp70 + tmp69
        _tmp70 = tl.where(rmask, tmp71, _tmp70)
        tmp72 = tl.load(in_ptr2 + load_seed_offset)
        tmp73 = r3 + (128*x4)
        tmp74 = tl.rand(tmp72, (tmp73).to(tl.uint32))
        tl.store(out_ptr1 + (r3 + (128*x4)), tmp67, rmask)
        tl.store(out_ptr3 + (r3 + (128*x4)), tmp74, rmask)
    tmp70 = tl.sum(_tmp70, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp75 = tl.load(out_ptr3 + (r3 + (128*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp79 = tl.load(out_ptr1 + (r3 + (128*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp76 = tmp75.to(tl.float32)
        tmp77 = 0.1
        tmp78 = tmp76 > tmp77
        tmp80 = tl.exp(tmp79)
        tmp81 = tmp80 / tmp70
        tmp82 = tmp78.to(tl.float32)
        tmp83 = tmp81.to(tl.float32)
        tmp84 = tmp82 * tmp83
        tmp85 = 1.1111111111111112
        tmp86 = tmp84 * tmp85
        tl.store(out_ptr4 + (r3 + (128*x4)), tmp78, rmask)
        tl.store(out_ptr5 + (r3 + (128*x4)), tmp81, rmask)
        tl.store(out_ptr6 + (r3 + (128*x4)), tmp86, rmask)
''')
