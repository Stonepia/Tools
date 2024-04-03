

# Original file: ./T5Small__0_forward_169.0/T5Small__0_forward_169.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/rl/crlshm3ik33krtvhbg3rxo2ekk4q5xc4chnm5dnqow3ntkefyutp.py
# Source Nodes: [dropout, softmax], Original ATen: [aten._softmax, aten.native_dropout]
# dropout => gt_2, mul_7, mul_8
# softmax => amax, div_2, exp, sub_2, sum_1
triton_red_fused__softmax_native_dropout_4 = async_compile.triton('triton_red_fused__softmax_native_dropout_4', '''
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
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_native_dropout_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_native_dropout_4(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024) % 8
    _tmp30 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (1024*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = r3 + ((-1)*x0)
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
        tmp26 = 0.0
        tmp27 = tmp25 + tmp26
        tmp28 = tmp0 + tmp27
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = triton_helpers.maximum(_tmp30, tmp29)
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
    tmp30 = triton_helpers.max2(_tmp30, 1)[:, None]
    _tmp64 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp32 = tl.load(in_ptr0 + (r3 + (1024*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp33 = r3 + ((-1)*x0)
        tmp34 = tl.full([1, 1], 0, tl.int64)
        tmp35 = tmp33 > tmp34
        tmp36 = tmp35.to(tl.int64)
        tmp37 = tl.full([1, 1], 16, tl.int64)
        tmp38 = tmp36 * tmp37
        tmp39 = tmp38 + tmp34
        tmp40 = tl.abs(tmp33)
        tmp41 = tl.full([1, 1], 8, tl.int64)
        tmp42 = tmp40 < tmp41
        tmp43 = tmp40.to(tl.float32)
        tmp44 = 8.0
        tmp45 = tmp43 / tmp44
        tmp46 = tl.log(tmp45)
        tmp47 = 2.772588722239781
        tmp48 = tmp46 / tmp47
        tmp49 = tmp48 * tmp44
        tmp50 = tmp49.to(tl.int64)
        tmp51 = tmp50 + tmp41
        tmp52 = tl.full([1, 1], 15, tl.int64)
        tmp53 = triton_helpers.minimum(tmp51, tmp52)
        tmp54 = tl.where(tmp42, tmp40, tmp53)
        tmp55 = tmp39 + tmp54
        tmp56 = tl.where(tmp55 < 0, tmp55 + 32, tmp55)
        # tl.device_assert(((0 <= tmp56) & (tmp56 < 32)) | ~rmask, "index out of bounds: 0 <= tmp56 < 32")
        tmp57 = tl.load(in_ptr1 + (x1 + (8*tmp56)), rmask, eviction_policy='evict_last')
        tmp58 = 0.0
        tmp59 = tmp57 + tmp58
        tmp60 = tmp32 + tmp59
        tmp61 = tmp60 - tmp30
        tmp62 = tl.exp(tmp61)
        tmp63 = tl.broadcast_to(tmp62, [XBLOCK, RBLOCK])
        tmp65 = _tmp64 + tmp63
        _tmp64 = tl.where(rmask, tmp65, _tmp64)
        tmp66 = tl.load(in_ptr2 + load_seed_offset)
        tmp67 = r3 + (1024*x4)
        tmp68 = tl.rand(tmp66, (tmp67).to(tl.uint32))
        tl.store(out_ptr2 + (r3 + (1024*x4)), tmp68, rmask)
    tmp64 = tl.sum(_tmp64, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp69 = tl.load(out_ptr2 + (r3 + (1024*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp72 = tl.load(in_ptr0 + (r3 + (1024*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp70 = 0.1
        tmp71 = tmp69 > tmp70
        tmp73 = r3 + ((-1)*x0)
        tmp74 = tl.full([1, 1], 0, tl.int64)
        tmp75 = tmp73 > tmp74
        tmp76 = tmp75.to(tl.int64)
        tmp77 = tl.full([1, 1], 16, tl.int64)
        tmp78 = tmp76 * tmp77
        tmp79 = tmp78 + tmp74
        tmp80 = tl.abs(tmp73)
        tmp81 = tl.full([1, 1], 8, tl.int64)
        tmp82 = tmp80 < tmp81
        tmp83 = tmp80.to(tl.float32)
        tmp84 = 8.0
        tmp85 = tmp83 / tmp84
        tmp86 = tl.log(tmp85)
        tmp87 = 2.772588722239781
        tmp88 = tmp86 / tmp87
        tmp89 = tmp88 * tmp84
        tmp90 = tmp89.to(tl.int64)
        tmp91 = tmp90 + tmp81
        tmp92 = tl.full([1, 1], 15, tl.int64)
        tmp93 = triton_helpers.minimum(tmp91, tmp92)
        tmp94 = tl.where(tmp82, tmp80, tmp93)
        tmp95 = tmp79 + tmp94
        tmp96 = tl.where(tmp95 < 0, tmp95 + 32, tmp95)
        # tl.device_assert(((0 <= tmp96) & (tmp96 < 32)) | ~rmask, "index out of bounds: 0 <= tmp96 < 32")
        tmp97 = tl.load(in_ptr1 + (x1 + (8*tmp96)), rmask, eviction_policy='evict_first')
        tmp98 = 0.0
        tmp99 = tmp97 + tmp98
        tmp100 = tmp72 + tmp99
        tmp101 = tmp100 - tmp30
        tmp102 = tl.exp(tmp101)
        tmp103 = tmp102 / tmp64
        tmp104 = tmp71.to(tl.float32)
        tmp105 = tmp104 * tmp103
        tmp106 = 1.1111111111111112
        tmp107 = tmp105 * tmp106
        tl.store(out_ptr3 + (r3 + (1024*x4)), tmp71, rmask)
        tl.store(out_ptr4 + (r3 + (1024*x4)), tmp103, rmask)
        tl.store(out_ptr5 + (r3 + (1024*x4)), tmp107, rmask)
''')
