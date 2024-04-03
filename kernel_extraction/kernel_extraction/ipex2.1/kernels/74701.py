

# Original file: ./maml__29_backward_89.15/maml__29_backward_89.15_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/rr/crrfhj3kkamyxvgsgtzmll3bjthjrwice2at4h6grq6r4wpeq6m6.py
# Source Nodes: [batch_norm], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
# batch_norm => convert_element_type_3
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_11 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_11', '''
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
    size_hints=[128, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (x1*((1 + (169*ks0)) // 2))
        tmp1 = 169*ks0
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((169*x0) + (10816*(((r2 + (x1*((1 + (169*ks0)) // 2))) // 169) % ks0)) + ((r2 + (x1*((1 + (169*ks0)) // 2))) % 169)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.where(tmp2, tmp4, 0)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp9 = tl.load(in_ptr1 + ((169*x0) + (10816*(((r2 + (x1*((1 + (169*ks0)) // 2))) // 169) % ks0)) + ((r2 + (x1*((1 + (169*ks0)) // 2))) % 169)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp4 * tmp12
        tmp14 = tl.where(tmp2, tmp13, 0)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp16, xmask)
''')
