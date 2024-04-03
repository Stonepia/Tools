

# Original file: ./hf_T5_large___60.0/hf_T5_large___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/m2/cm24fgbbzjpqd26mlalepzifgupr6u42cxezxfvafz452vjpze5x.py
# Source Nodes: [float_2, softmax, type_as], Original ATen: [aten._softmax, aten._to_copy]
# float_2 => convert_element_type_10
# softmax => amax, div_2, exp, sub_2, sum_1
# type_as => convert_element_type_11
triton_red_fused__softmax__to_copy_5 = async_compile.triton('triton_red_fused__softmax__to_copy_5', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_5(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp31 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp26 = tl.load(in_ptr1 + (x1 + (16*tmp25)), rmask, eviction_policy='evict_last')
        tmp27 = tmp1 + tmp26
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = triton_helpers.maximum(_tmp31, tmp30)
        _tmp31 = tl.where(rmask, tmp32, _tmp31)
    tmp31 = triton_helpers.max2(_tmp31, 1)[:, None]
    _tmp66 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp33 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp34 = tmp33.to(tl.float32)
        tmp35 = r2 + ((-1)*x0)
        tmp36 = tl.full([1, 1], 0, tl.int64)
        tmp37 = tmp35 > tmp36
        tmp38 = tmp37.to(tl.int64)
        tmp39 = tl.full([1, 1], 16, tl.int64)
        tmp40 = tmp38 * tmp39
        tmp41 = tmp40 + tmp36
        tmp42 = tl.abs(tmp35)
        tmp43 = tl.full([1, 1], 8, tl.int64)
        tmp44 = tmp42 < tmp43
        tmp45 = tmp42.to(tl.float32)
        tmp46 = 8.0
        tmp47 = tmp45 / tmp46
        tmp48 = tl.log(tmp47)
        tmp49 = 2.772588722239781
        tmp50 = tmp48 / tmp49
        tmp51 = tmp50 * tmp46
        tmp52 = tmp51.to(tl.int64)
        tmp53 = tmp52 + tmp43
        tmp54 = tl.full([1, 1], 15, tl.int64)
        tmp55 = triton_helpers.minimum(tmp53, tmp54)
        tmp56 = tl.where(tmp44, tmp42, tmp55)
        tmp57 = tmp41 + tmp56
        tmp58 = tl.where(tmp57 < 0, tmp57 + 32, tmp57)
        # tl.device_assert(((0 <= tmp58) & (tmp58 < 32)) | ~rmask, "index out of bounds: 0 <= tmp58 < 32")
        tmp59 = tl.load(in_ptr1 + (x1 + (16*tmp58)), rmask, eviction_policy='evict_last')
        tmp60 = tmp34 + tmp59
        tmp61 = tmp60.to(tl.float32)
        tmp62 = tmp61.to(tl.float32)
        tmp63 = tmp62 - tmp31
        tmp64 = tl.exp(tmp63)
        tmp65 = tl.broadcast_to(tmp64, [XBLOCK, RBLOCK])
        tmp67 = _tmp66 + tmp65
        _tmp66 = tl.where(rmask, tmp67, _tmp66)
    tmp66 = tl.sum(_tmp66, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp68 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp69 = tmp68.to(tl.float32)
        tmp70 = r2 + ((-1)*x0)
        tmp71 = tl.full([1, 1], 0, tl.int64)
        tmp72 = tmp70 > tmp71
        tmp73 = tmp72.to(tl.int64)
        tmp74 = tl.full([1, 1], 16, tl.int64)
        tmp75 = tmp73 * tmp74
        tmp76 = tmp75 + tmp71
        tmp77 = tl.abs(tmp70)
        tmp78 = tl.full([1, 1], 8, tl.int64)
        tmp79 = tmp77 < tmp78
        tmp80 = tmp77.to(tl.float32)
        tmp81 = 8.0
        tmp82 = tmp80 / tmp81
        tmp83 = tl.log(tmp82)
        tmp84 = 2.772588722239781
        tmp85 = tmp83 / tmp84
        tmp86 = tmp85 * tmp81
        tmp87 = tmp86.to(tl.int64)
        tmp88 = tmp87 + tmp78
        tmp89 = tl.full([1, 1], 15, tl.int64)
        tmp90 = triton_helpers.minimum(tmp88, tmp89)
        tmp91 = tl.where(tmp79, tmp77, tmp90)
        tmp92 = tmp76 + tmp91
        tmp93 = tl.where(tmp92 < 0, tmp92 + 32, tmp92)
        # tl.device_assert(((0 <= tmp93) & (tmp93 < 32)) | ~rmask, "index out of bounds: 0 <= tmp93 < 32")
        tmp94 = tl.load(in_ptr1 + (x1 + (16*tmp93)), rmask, eviction_policy='evict_first')
        tmp95 = tmp69 + tmp94
        tmp96 = tmp95.to(tl.float32)
        tmp97 = tmp96.to(tl.float32)
        tmp98 = tmp97 - tmp31
        tmp99 = tl.exp(tmp98)
        tmp100 = tmp99 / tmp66
        tmp101 = tmp100.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (512*x3)), tmp101, rmask)
''')
