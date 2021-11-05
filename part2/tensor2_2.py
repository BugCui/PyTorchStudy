import torch

# Tensor 操作
if __name__ == '__main__':

    # 创建tensor

    x = torch.empty(5,3)
    print(x)

    print(" ")

    x = torch.rand(5,3)
    print(x)

    print("")

    x = torch.zeros(5, 3, dtype=torch.long)
    print(x)

    print("")

    x = torch.tensor([5.5,3])
    print(x)

    print("")

    x = x.new_ones(5, 3, dtype=torch.float64) # 返回的tensor默认具有相同的torch.dtype和torch.device
    print(x)

    x = torch.randn_like(x, dtype=torch.float) # 指定新的数据类型
    print(x)

    print("")

    print(x.size())
    print(x.shape)

    print("")

    # 算数操作
    y = torch.rand(5, 3)
    print(x)
    print(y)
    print(x + y)

    print("")

    print(torch.add(x, y))

    print("")

    result = torch.empty(5,3)
    torch.add(x, y, out=result)
    print(result)

    print("")

    y.add_(x)
    print(y)

    print()

    # 索引
    print(y)
    print(x)
    y = x[0, :]
    y += 1
    print(y)
    print(x[0, :]) # 源tensor也被改了

    print()

    # 改变形状
    y = x.view(15)
    print(y)
    print(x)
    z = x.view(-1, 5) # -1所指的维度可以根据其他维度的值推出来
    print(z)
    print(x.size(), y.size(), z.size())

    print()

    x += 1
    print(x)
    print(y) # y也加了1

    print()

    x_cp = x.clone().view(15) # 深拷贝
    x -= 1
    print(x)
    print(x_cp)

    print()

    x = torch.randn(1)
    print(x)
    print(x.item())

    # 线性代数

    print()

    # 广播机制
    # 1 2   1 1   2 3
    # 1 2 + 2 2 = 3 4
    # 1 2   3 3   4 5
    #
    x = torch.arange(1, 3).view(1, 2)
    print(x)
    y = torch.arange(1, 4).view(3, 1)
    print(y)
    print(x + y)

    print()

    # 运算的内存开销
    x = torch.tensor([1,2])
    y = torch.tensor([3,4])
    id_before = id(y)
    y = y + x
    print(y)
    print(id(y) == id_before) # False 表示 y = y + x 会开辟新内存

    print()

    x = torch.tensor([1,2])
    y = torch.tensor([3,4])
    id_before = id(y)
    y[:] = y + x
    print(y)
    print(id(y) == id_before) # True

    print()

    x = torch.tensor([1,2])
    y = torch.tensor([3,4])
    id_before = id(y)
    torch.add(x, y, out=y)
    print(y)
    print(id(y) == id_before) # True

    # 注：虽然view返回的Tensor与源Tensor是共享data的，但是依然是一个新的Tensor（因为Tensor除了包含data外还有一些其他属性），二者id（内存地址）并不一致。

    print()

    # Tensor 和 NumPy 相互转换
    a = torch.ones(5)
    b = a.numpy()
    print(a, b)

    a += 1
    print(a, b)
    b += 1
    print(a, b)

    print("")

    c = torch.tensor(a)
    a += 1
    print(a, c)

    print()

    # Tensor on GPU
    # 以下代码只有在PyTorch GPU版本上才会执行
    if torch.cuda.is_available():
        device = torch.device("cuda")  # GPU
        y = torch.ones_like(x, device=device)  # 直接创建一个在GPU上的Tensor
        x = x.to(device)  # 等价于 .to("cuda")
        z = x + y
        print(z)
        print(z.to("cpu", torch.double))  # to()还可以同时更改数据类型


