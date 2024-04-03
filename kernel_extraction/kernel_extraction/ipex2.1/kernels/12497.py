

# Original file: ./MobileBertForMaskedLM__0_forward_208.0/MobileBertForMaskedLM__0_forward_208.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/sv/csvpn3nd2q2trla4tlqbmxosc6uw25osldqymxznvjvvzrqalnje.py
# Source Nodes: [cross_entropy, iadd], Original ATen: [aten._log_softmax, aten.view]
# cross_entropy => amax_24, convert_element_type_51, convert_element_type_52, exp_24, log, sub_26, sub_27, sum_25
# iadd => view_967
triton_red_fused__log_softmax_view_21 = async_compile.triton('triton_red_fused__log_softmax_view_21', '''
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
    size_hints=[16384, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_view_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__log_softmax_view_21(in_out_ptr0, in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 30522
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r1 + (30522*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = triton_helpers.maximum(_tmp5, tmp4)
        _tmp5 = tl.where(rmask, tmp6, _tmp5)
        tl.store(in_out_ptr0 + (r1 + (30522*x0)), tmp2, rmask)
    tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(in_out_ptr0 + (r1 + (30522*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 - tmp5
        tmp10 = tl.exp(tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_out_ptr0 + (r1 + (30522*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 - tmp5
        tmp17 = tl.log(tmp12)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp18.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (30522*x0)), tmp19, rmask)
''')
