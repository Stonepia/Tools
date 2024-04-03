

# Original file: ./MT5ForConditionalGeneration__0_forward_205.0/MT5ForConditionalGeneration__0_forward_205.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/k6/ck64ac3ydf2qamk2okh2rkqzl74t4p3sc272dgwzs62uubk2o77e.py
# Source Nodes: [dropout_8, softmax_8], Original ATen: [aten._softmax, aten.native_dropout]
# dropout_8 => gt_36, mul_153, mul_154
# softmax_8 => amax_8, div_12, exp_8, sub_13, sum_9
triton_red_fused__softmax_native_dropout_11 = async_compile.triton('triton_red_fused__softmax_native_dropout_11', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_native_dropout_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_native_dropout_11(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr3, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128) % 6
    _tmp31 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = (-1)*(tl.minimum(0, r3 + ((-1)*x0), tl.PropagateNan.NONE))
        tmp2 = tl.full([1, 1], 16, tl.int64)
        tmp3 = tmp1 < tmp2
        tmp4 = tmp1.to(tl.float32)
        tmp5 = 16.0
        tmp6 = tmp4 / tmp5
        tmp7 = tl.log(tmp6)
        tmp8 = 2.0794415416798357
        tmp9 = tmp7 / tmp8
        tmp10 = tmp9 * tmp5
        tmp11 = tmp10.to(tl.int64)
        tmp12 = tmp11 + tmp2
        tmp13 = tl.full([1, 1], 31, tl.int64)
        tmp14 = triton_helpers.minimum(tmp12, tmp13)
        tmp15 = tl.where(tmp3, tmp1, tmp14)
        tmp16 = tl.full([1, 1], 0, tl.int64)
        tmp17 = tmp15 + tmp16
        tmp18 = tl.where(tmp17 < 0, tmp17 + 32, tmp17)
        # tl.device_assert((0 <= tmp18) & (tmp18 < 32), "index out of bounds: 0 <= tmp18 < 32")
        tmp19 = tl.load(in_ptr1 + (x1 + (6*tmp18)), None, eviction_policy='evict_last')
        tmp20 = r3
        tmp21 = x0
        tmp22 = tmp20 <= tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = 1.0
        tmp25 = tmp24 - tmp23
        tmp26 = -3.4028234663852886e+38
        tmp27 = tmp25 * tmp26
        tmp28 = tmp19 + tmp27
        tmp29 = tmp0 + tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = triton_helpers.maximum(_tmp31, tmp30)
        _tmp31 = tl.where(rmask, tmp32, _tmp31)
    tmp31 = triton_helpers.max2(_tmp31, 1)[:, None]
    _tmp66 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp33 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp34 = (-1)*(tl.minimum(0, r3 + ((-1)*x0), tl.PropagateNan.NONE))
        tmp35 = tl.full([1, 1], 16, tl.int64)
        tmp36 = tmp34 < tmp35
        tmp37 = tmp34.to(tl.float32)
        tmp38 = 16.0
        tmp39 = tmp37 / tmp38
        tmp40 = tl.log(tmp39)
        tmp41 = 2.0794415416798357
        tmp42 = tmp40 / tmp41
        tmp43 = tmp42 * tmp38
        tmp44 = tmp43.to(tl.int64)
        tmp45 = tmp44 + tmp35
        tmp46 = tl.full([1, 1], 31, tl.int64)
        tmp47 = triton_helpers.minimum(tmp45, tmp46)
        tmp48 = tl.where(tmp36, tmp34, tmp47)
        tmp49 = tl.full([1, 1], 0, tl.int64)
        tmp50 = tmp48 + tmp49
        tmp51 = tl.where(tmp50 < 0, tmp50 + 32, tmp50)
        # tl.device_assert((0 <= tmp51) & (tmp51 < 32), "index out of bounds: 0 <= tmp51 < 32")
        tmp52 = tl.load(in_ptr1 + (x1 + (6*tmp51)), None, eviction_policy='evict_first')
        tmp53 = r3
        tmp54 = x0
        tmp55 = tmp53 <= tmp54
        tmp56 = tmp55.to(tl.float32)
        tmp57 = 1.0
        tmp58 = tmp57 - tmp56
        tmp59 = -3.4028234663852886e+38
        tmp60 = tmp58 * tmp59
        tmp61 = tmp52 + tmp60
        tmp62 = tmp33 + tmp61
        tmp63 = tmp62 - tmp31
        tmp64 = tl.exp(tmp63)
        tmp65 = tl.broadcast_to(tmp64, [XBLOCK, RBLOCK])
        tmp67 = _tmp66 + tmp65
        _tmp66 = tl.where(rmask, tmp67, _tmp66)
        tmp68 = tl.load(in_ptr2 + load_seed_offset)
        tmp69 = r3 + (128*x4)
        tmp70 = tl.rand(tmp68, (tmp69).to(tl.uint32))
        tl.store(out_ptr1 + (r3 + (128*x4)), tmp63, rmask)
        tl.store(out_ptr3 + (r3 + (128*x4)), tmp70, rmask)
    tmp66 = tl.sum(_tmp66, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp71 = tl.load(out_ptr3 + (r3 + (128*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp74 = tl.load(out_ptr1 + (r3 + (128*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp72 = 0.1
        tmp73 = tmp71 > tmp72
        tmp75 = tl.exp(tmp74)
        tmp76 = tmp75 / tmp66
        tmp77 = tmp73.to(tl.float32)
        tmp78 = tmp77 * tmp76
        tmp79 = 1.1111111111111112
        tmp80 = tmp78 * tmp79
        tl.store(out_ptr4 + (r3 + (128*x4)), tmp73, rmask)
        tl.store(out_ptr5 + (r3 + (128*x4)), tmp76, rmask)
        tl.store(out_ptr6 + (r3 + (128*x4)), tmp80, rmask)
''')
