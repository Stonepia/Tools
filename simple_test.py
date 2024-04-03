import torch
import intel_extension_for_pytorch
# torch.xpu.set_log_level(0)
class Test(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x+y
        return z

def make_input(device):
    torch.manual_seed(1337)
    return torch.randn((2, 3, 240, 240), dtype=torch.float32, device='cpu').to(device), torch.randn((2, 3, 240, 240), dtype=torch.float32, device='cpu').to(device)

def check_model(device):
    model = Test().to(device)
    x, y = make_input(device)
    # torch.xpu.set_log_level(0)
    res1 = model(x, y)
    
    model = torch.compile(model, backend='inductor')
    res2 = model(x, y)

    print(f"{device} results match: {torch.allclose(res1, res2, atol=1e-4, rtol=1e-4)}")

    return res1, res2

# cpu1, cpu2 = check_model('cpu')
xpu1, xpu2 = check_model('xpu')

# print(f"Eager results match: {torch.allclose(cpu1, xpu1.to('cpu'), atol=1e-4, rtol=1e-4)}")
# torch.testing.assert_close(cpu1, xpu1.to('cpu'), atol=1e-4, rtol=1e-4)


# print(f"Inductor results match: {torch.allclose(cpu2, xpu2.to('cpu'), atol=1e-4, rtol=1e-4)}")


# class PrecompiledPatternMatcherPass(object):
#     def __init__(self):
#         super().__init__()

#     def __call__(self, g: torch.fx.graph.Graph):
#         self.apply(g)
    
#     def __repr__(self):
#         return f"{self.__}"

# cls = PrecompiledPatternMatcherPass()
# print(f"class is {cls}")