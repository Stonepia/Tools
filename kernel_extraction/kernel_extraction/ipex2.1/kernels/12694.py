

# Original file: ./hf_Reformer__25_inference_65.5/hf_Reformer__25_inference_65.5_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/nd/cndoz4px73yhz4lln5ie3rwe5jesxyqbp2fhup5232ywyrv7ssud.py
# Source Nodes: [exp, logsumexp, sub], Original ATen: [aten.exp, aten.logsumexp, aten.sub]
# exp => exp_1
# logsumexp => abs_1, add_2, amax, convert_element_type_7, convert_element_type_8, eq, exp, full_default_1, log, sub_1, sum_1, where
# sub => sub_2
triton_per_fused_exp_logsumexp_sub_7 = async_compile.triton('triton_per_fused_exp_logsumexp_sub_7', '''
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
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_exp_logsumexp_sub_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_exp_logsumexp_sub_7(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask, tmp2, float("-inf"))
    tmp5 = triton_helpers.max2(tmp4, 1)[:, None]
    tmp6 = tl.abs(tmp5)
    tmp7 = float("inf")
    tmp8 = tmp6 == tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = tmp1 - tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tl.log(tmp16)
    tmp18 = tmp17 + tmp10
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp0 - tmp19
    tmp21 = tl.exp(tmp20)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp21, rmask)
''')
