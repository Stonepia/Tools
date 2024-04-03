

# Original file: ./hf_T5_generate__81_inference_121.61/hf_T5_generate__81_inference_121.61_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/vr/cvriuolpnayi36ziaqbqrjysfo5sf473mobhkxoro2dgr5giurx3.py
# Source Nodes: [float_3, softmax_1, type_as_1], Original ATen: [aten._softmax, aten._to_copy]
# float_3 => convert_element_type_17
# softmax_1 => amax_1, div_3, exp_1, sub_9, sum_2
# type_as_1 => convert_element_type_18
triton_red_fused__softmax__to_copy_8 = async_compile.triton('triton_red_fused__softmax__to_copy_8', '''
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
    size_hints=[8, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*i64', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_8(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 1.0
        tmp5 = tmp4 - tmp3
        tmp6 = -3.4028234663852886e+38
        tmp7 = tmp5 * tmp6
        tmp8 = 0.0
        tmp9 = tmp8 + tmp7
        tmp10 = tmp1 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = triton_helpers.maximum(_tmp14, tmp13)
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = triton_helpers.max2(_tmp14, 1)[:, None]
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp16.to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = 1.0
        tmp21 = tmp20 - tmp19
        tmp22 = -3.4028234663852886e+38
        tmp23 = tmp21 * tmp22
        tmp24 = 0.0
        tmp25 = tmp24 + tmp23
        tmp26 = tmp17 + tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 - tmp14
        tmp30 = tl.exp(tmp29)
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask & xmask, tmp33, _tmp32)
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp34 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp36 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tmp34.to(tl.float32)
        tmp37 = tmp36.to(tl.float32)
        tmp38 = 1.0
        tmp39 = tmp38 - tmp37
        tmp40 = -3.4028234663852886e+38
        tmp41 = tmp39 * tmp40
        tmp42 = 0.0
        tmp43 = tmp42 + tmp41
        tmp44 = tmp35 + tmp43
        tmp45 = tmp44.to(tl.float32)
        tmp46 = tmp45.to(tl.float32)
        tmp47 = tmp46 - tmp14
        tmp48 = tl.exp(tmp47)
        tmp49 = tmp48 / tmp32
        tmp50 = tmp49.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp50, rmask & xmask)
''')
