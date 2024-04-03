

# Original file: ./AllenaiLongformerBase__22_backward_143.5/AllenaiLongformerBase__22_backward_143.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/xl/cxl2lbjh4xeghedi3hqvqa66yfjpkdo4fmtv4g3fvudv4dhrxbcg.py
# Source Nodes: [tril], Original ATen: [aten._softmax_backward_data, aten.clone, aten.masked_fill, aten.native_dropout_backward, aten.tril]
# tril => full_default_1
triton_red_fused__softmax_backward_data_clone_masked_fill_native_dropout_backward_tril_14 = async_compile.triton('triton_red_fused__softmax_backward_data_clone_masked_fill_native_dropout_backward_tril_14', '''
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
    size_hints=[65536, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_backward_data_clone_masked_fill_native_dropout_backward_tril_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_backward_data_clone_masked_fill_native_dropout_backward_tril_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 513
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = (xindex // 12)
    tmp0 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    x1 = (xindex // 12) % 1024
    x0 = xindex % 12
    x2 = (xindex // 12288)
    x5 = xindex
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp19 = tl.load(in_ptr2 + (r3 + (513*x5)), rmask, eviction_policy='evict_last')
        tmp25 = tl.load(in_ptr3 + (r3 + (513*x5)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = r3
        tmp2 = tl.full([1, 1], 770, tl.int64)
        tmp3 = tmp1 < tmp2
        tmp4 = r3 + (770*(x1 % 256))
        tmp5 = tl.full([1, 1], 196864, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tmp6 & tmp3
        tmp8 = (r3 + (770*(x1 % 256))) % 769
        tmp9 = tl.full([1, 1], 768, tl.int64)
        tmp10 = tmp8 < tmp9
        tmp11 = tmp10 & tmp7
        tmp12 = tl.load(in_ptr1 + ((768*(((r3 + (770*(x1 % 256))) // 769) % 256)) + (196608*(x1 // 256)) + (786432*x0) + (9437184*x2) + ((r3 + (770*(x1 % 256))) % 769)), rmask & tmp11, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.where(tmp11, tmp12, 0.0)
        tmp14 = 0.0
        tmp15 = tl.where(tmp10, tmp13, tmp14)
        tmp16 = tl.where(tmp7, tmp15, 0.0)
        tmp17 = tl.where(tmp6, tmp16, tmp14)
        tmp18 = tl.where(tmp3, tmp17, 0.0)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = 1.1111111111111112
        tmp22 = tmp20 * tmp21
        tmp23 = tmp18 * tmp22
        tmp24 = tl.where(tmp0, tmp14, tmp23)
        tmp26 = tmp24 * tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask, tmp29, _tmp28)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp48 = tl.load(in_ptr2 + (r3 + (513*x5)), rmask, eviction_policy='evict_first')
        tmp54 = tl.load(in_ptr3 + (r3 + (513*x5)), rmask, eviction_policy='evict_first', other=0.0)
        tmp30 = r3
        tmp31 = tl.full([1, 1], 770, tl.int64)
        tmp32 = tmp30 < tmp31
        tmp33 = r3 + (770*(x1 % 256))
        tmp34 = tl.full([1, 1], 196864, tl.int64)
        tmp35 = tmp33 < tmp34
        tmp36 = tmp35 & tmp32
        tmp37 = (r3 + (770*(x1 % 256))) % 769
        tmp38 = tl.full([1, 1], 768, tl.int64)
        tmp39 = tmp37 < tmp38
        tmp40 = tmp39 & tmp36
        tmp41 = tl.load(in_ptr1 + ((768*(((r3 + (770*(x1 % 256))) // 769) % 256)) + (196608*(x1 // 256)) + (786432*x0) + (9437184*x2) + ((r3 + (770*(x1 % 256))) % 769)), rmask & tmp40, eviction_policy='evict_first', other=0.0)
        tmp42 = tl.where(tmp40, tmp41, 0.0)
        tmp43 = 0.0
        tmp44 = tl.where(tmp39, tmp42, tmp43)
        tmp45 = tl.where(tmp36, tmp44, 0.0)
        tmp46 = tl.where(tmp35, tmp45, tmp43)
        tmp47 = tl.where(tmp32, tmp46, 0.0)
        tmp49 = tmp48.to(tl.float32)
        tmp50 = 1.1111111111111112
        tmp51 = tmp49 * tmp50
        tmp52 = tmp47 * tmp51
        tmp53 = tl.where(tmp0, tmp43, tmp52)
        tmp55 = tmp53 * tmp54
        tmp56 = tmp54 * tmp28
        tmp57 = tmp55 - tmp56
        tl.store(out_ptr1 + (r3 + (513*x5)), tmp57, rmask)
''')
