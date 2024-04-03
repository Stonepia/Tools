

# Original file: ./maml__29_backward_89.15/maml__29_backward_89.15_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/pl/cplfn2rpcbiu5vi3xoerzthqeud4qzx6ezjm22o6bfldfbklpmpb.py
# Source Nodes: [batch_norm_3, relu_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.relu]
# batch_norm_3 => convert_element_type_12
# relu_3 => relu_3
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_1 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_1', '''
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
    size_hints=[64, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask & xmask, tmp4, _tmp3)
        tmp6 = triton_helpers.maximum(0, tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp1 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp12, xmask)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tmp12 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp3.to(tl.float32)
    tl.store(out_ptr2 + (x0), tmp16, xmask)
    tl.store(out_ptr3 + (x0), tmp17, xmask)
''')