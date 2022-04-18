from cnn_gp import Conv2d,ReLU,Sequential,Sum



def cnn(n_layers,filter_size=7,var_b=7.86,var_w=2.79,stride=1):
    layers=[]
    for i in range(n_layers):
        layers+=[Conv2d(kernel_size=filter_size,padding='same',var_weight=var_w*filter_size**2,var_bias=var_b,stride=stride),ReLU()]
    return Sequential(*layers,Conv2d(kernel_size=28,padding=0,var_weight=var_w,var_bias=var_b))


def res_cnn(n_layers,filter_size=4,var_b=4.69,var_w=7.27,stride=1):
    model = Sequential(
        *(Sum([
            Sequential(),
            Sequential(
                Conv2d(kernel_size=filter_size, padding="same", var_weight=var_w * filter_size**2,var_bias=var_b,stride=stride),
                ReLU()
            )]) for _ in range(n_layers-1)),
        Conv2d(kernel_size=filter_size, padding="same", var_weight=var_w * filter_size**2,var_bias=var_b),
        ReLU(),
        Conv2d(kernel_size=28, padding=0, var_weight=var_w,var_bias=var_b))
    return model