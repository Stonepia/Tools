

# Original file: ./maml__21_forward_62.1/maml__21_forward_62.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/tx/ctx24loavakc6nwf4bko7tjga52eik6orvnkoe5xamtijl2u2b7w.py
# Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_forward]
# cross_entropy => convert_element_type_18, div, full_default_1, ne, neg, sum_2, sum_3, where_1
triton_per_fused_nll_loss_forward_14 = async_compile.triton('triton_per_fused_nll_loss_forward_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[1, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_nll_loss_forward_14(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1, 1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([1, 1], 0, tl.int64)
    tmp9 = tl.where(tmp2, tmp0, tmp8)
    tmp10 = tl.where(tmp9 < 0, tmp9 + 5, tmp9)
    # tl.device_assert((0 <= tmp10) & (tmp10 < 5), "index out of bounds: 0 <= tmp10 < 5")
    tmp11 = tl.load(in_ptr1 + (tmp10 + (5*r0)), rmask, other=0.0).to(tl.float32)
    tmp12 = -tmp11
    tmp13 = 0.0
    tmp14 = tl.where(tmp2, tmp12, tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp7.to(tl.float32)
    tmp20 = tmp18 / tmp19
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp19, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp20, None)
''')