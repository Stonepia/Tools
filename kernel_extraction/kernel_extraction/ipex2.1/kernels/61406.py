

# Original file: ./MT5ForConditionalGeneration__0_forward_205.0/MT5ForConditionalGeneration__0_forward_205.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/6j/c6j4tzassxjydbcvqz3j76jmob4k2m52fise3rxzpeghozkezkfk.py
# Source Nodes: [dropout, float_2, softmax, type_as], Original ATen: [aten._softmax, aten._to_copy, aten.native_dropout]
# dropout => gt_2, mul_7, mul_8
# float_2 => convert_element_type_6
# softmax => amax, div_2, exp, sub_2, sum_1
# type_as => convert_element_type_7
triton_red_fused__softmax__to_copy_native_dropout_4 = async_compile.triton('triton_red_fused__softmax__to_copy_native_dropout_4', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*i64', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp16', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_native_dropout_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_native_dropout_4(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp25 = tl.load(in_ptr1 + (x1 + (6*tmp24)), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp26 = 0.0
        tmp27 = tmp25 + tmp26
        tmp28 = tmp0 + tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = triton_helpers.maximum(_tmp31, tmp30)
        _tmp31 = tl.where(rmask, tmp32, _tmp31)
    tmp31 = triton_helpers.max2(_tmp31, 1)[:, None]
    _tmp66 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp33 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp34 = r3 + ((-1)*x0)
        tmp35 = tl.full([1, 1], 0, tl.int64)
        tmp36 = tmp34 > tmp35
        tmp37 = tmp36.to(tl.int64)
        tmp38 = tl.full([1, 1], 16, tl.int64)
        tmp39 = tmp37 * tmp38
        tmp40 = tmp39 + tmp35
        tmp41 = tl.abs(tmp34)
        tmp42 = tl.full([1, 1], 8, tl.int64)
        tmp43 = tmp41 < tmp42
        tmp44 = tmp41.to(tl.float32)
        tmp45 = 8.0
        tmp46 = tmp44 / tmp45
        tmp47 = tl.log(tmp46)
        tmp48 = 2.772588722239781
        tmp49 = tmp47 / tmp48
        tmp50 = tmp49 * tmp45
        tmp51 = tmp50.to(tl.int64)
        tmp52 = tmp51 + tmp42
        tmp53 = tl.full([1, 1], 15, tl.int64)
        tmp54 = triton_helpers.minimum(tmp52, tmp53)
        tmp55 = tl.where(tmp43, tmp41, tmp54)
        tmp56 = tmp40 + tmp55
        tmp57 = tl.where(tmp56 < 0, tmp56 + 32, tmp56)
        # tl.device_assert(((0 <= tmp57) & (tmp57 < 32)) | ~rmask, "index out of bounds: 0 <= tmp57 < 32")
        tmp58 = tl.load(in_ptr1 + (x1 + (6*tmp57)), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp59 = 0.0
        tmp60 = tmp58 + tmp59
        tmp61 = tmp33 + tmp60
        tmp62 = tmp61.to(tl.float32)
        tmp63 = tmp62 - tmp31
        tmp64 = tl.exp(tmp63)
        tmp65 = tl.broadcast_to(tmp64, [XBLOCK, RBLOCK])
        tmp67 = _tmp66 + tmp65
        _tmp66 = tl.where(rmask, tmp67, _tmp66)
        tmp68 = tl.load(in_ptr2 + load_seed_offset)
        tmp69 = r3 + (128*x4)
        tmp70 = tl.rand(tmp68, (tmp69).to(tl.uint32))
        tl.store(out_ptr2 + (r3 + (128*x4)), tmp70, rmask)
    tmp66 = tl.sum(_tmp66, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp71 = tl.load(out_ptr2 + (r3 + (128*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp75 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp72 = tmp71.to(tl.float32)
        tmp73 = 0.1
        tmp74 = tmp72 > tmp73
        tmp76 = r3 + ((-1)*x0)
        tmp77 = tl.full([1, 1], 0, tl.int64)
        tmp78 = tmp76 > tmp77
        tmp79 = tmp78.to(tl.int64)
        tmp80 = tl.full([1, 1], 16, tl.int64)
        tmp81 = tmp79 * tmp80
        tmp82 = tmp81 + tmp77
        tmp83 = tl.abs(tmp76)
        tmp84 = tl.full([1, 1], 8, tl.int64)
        tmp85 = tmp83 < tmp84
        tmp86 = tmp83.to(tl.float32)
        tmp87 = 8.0
        tmp88 = tmp86 / tmp87
        tmp89 = tl.log(tmp88)
        tmp90 = 2.772588722239781
        tmp91 = tmp89 / tmp90
        tmp92 = tmp91 * tmp87
        tmp93 = tmp92.to(tl.int64)
        tmp94 = tmp93 + tmp84
        tmp95 = tl.full([1, 1], 15, tl.int64)
        tmp96 = triton_helpers.minimum(tmp94, tmp95)
        tmp97 = tl.where(tmp85, tmp83, tmp96)
        tmp98 = tmp82 + tmp97
        tmp99 = tl.where(tmp98 < 0, tmp98 + 32, tmp98)
        # tl.device_assert(((0 <= tmp99) & (tmp99 < 32)) | ~rmask, "index out of bounds: 0 <= tmp99 < 32")
        tmp100 = tl.load(in_ptr1 + (x1 + (6*tmp99)), rmask, eviction_policy='evict_first').to(tl.float32)
        tmp101 = 0.0
        tmp102 = tmp100 + tmp101
        tmp103 = tmp75 + tmp102
        tmp104 = tmp103.to(tl.float32)
        tmp105 = tmp104 - tmp31
        tmp106 = tl.exp(tmp105)
        tmp107 = tmp106 / tmp66
        tmp108 = tmp74.to(tl.float32)
        tmp109 = tmp107.to(tl.float32)
        tmp110 = tmp108 * tmp109
        tmp111 = 1.1111111111111112
        tmp112 = tmp110 * tmp111
        tl.store(out_ptr3 + (r3 + (128*x4)), tmp74, rmask)
        tl.store(out_ptr4 + (r3 + (128*x4)), tmp107, rmask)
        tl.store(out_ptr5 + (r3 + (128*x4)), tmp112, rmask)
''')
