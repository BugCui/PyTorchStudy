import torch

if __name__ == '__main__':

    # Tensor

    x = torch.ones(2, 2, requires_grad=True)
    print(x)
    print(x.grad_fn)

    print()

    y = x + 2
    print(y)
    print(y.grad_fn)

    print()

    print(x.is_leaf, y.is_leaf)

    print()

    z  = y * y * 3
    out = z.mean()
    print(z, out)

    print()

    a = torch.randn(2, 2)
    a = ((a * 3) / (a - 1))
    print(a.requires_grad)
    a.requires_grad_(True)
    print(a.requires_grad)
    b = (a * a).sum()
    print(b.grad_fn)

    print()

    # 梯度
    print("######梯度##########")
    print(x.grad)
    out.backward()
    print(x)
    print(x.grad)

    print()

    # 再来反向传播一次，注意grad是累加的
    out2 = x.sum()
    out2.backward()
    print(x.grad)

    out3 = x.sum()
    x.grad.data.zero_()
    out3.backward()
    print(x.grad)

    print()


    x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    y = 2 * x
    z = y.view(2, 2,)
    print(z)

    print()

    v= torch.tensor([[1.0, 0.1],[0.01,0.001]], dtype=torch.float)
    z.backward(v)
    print(x.grad)

    print()

    x = torch.tensor(1.0, requires_grad=True)
    y1 = x ** 2
    with torch.no_grad():
        y2 = x ** 3
    y3 = y1 + y2

    print(x.requires_grad)
    print(y1, y1.requires_grad)
    print(y2, y2.requires_grad)
    print(y3, y3.requires_grad)

    print()

    y3.backward()
    print(x.grad)

    print()

    x = torch.ones(1, requires_grad=True)

    print(x.data)
    print(x.data.requires_grad)

    y = 2 * x
    x.data *= 100

    y.backward()
    print(x)
    print(x.grad)