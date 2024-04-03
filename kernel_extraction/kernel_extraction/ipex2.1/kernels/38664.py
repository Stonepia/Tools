

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/5o/c5oo44ryl4debfuoy5b273uiekpplr7fwmgqyctsrdcmkzx76ge6.py
# Source Nodes: [mse_loss_fn], Original ATen: [aten._to_copy, aten.mse_loss]
# mse_loss_fn => convert_element_type_354, convert_element_type_355, mean_1, pow_1, sub_110
triton_per_fused__to_copy_mse_loss_55 = async_compile.triton('triton_per_fused__to_copy_mse_loss_55', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_mse_loss_55', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_mse_loss_55(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 46464
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 192
    x1 = (xindex // 192)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((512*((r2 + (128*x0) + (24576*x1)) % 1936)) + (991232*((r2 + (128*x0) + (24576*x1)) // 991232)) + (((r2 + (128*x0) + (24576*x1)) // 1936) % 512)), rmask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((512*((r2 + (128*x0) + (24576*x1)) % 1936)) + (991232*((r2 + (128*x0) + (24576*x1)) // 991232)) + (((r2 + (128*x0) + (24576*x1)) // 1936) % 512)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 - tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')