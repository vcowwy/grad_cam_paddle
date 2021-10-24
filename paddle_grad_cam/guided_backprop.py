import numpy as np
import paddle

from paddle_grad_cam.utils.find_layers import replace_all_layer_type_recursive


class GuidedBackpropReLU(paddle.autograd.PyLayer):

    @staticmethod
    def forward(self, input_img):
        positive_mask = paddle.to_tensor(data=(input_img > 0), dtype=input_img.dtype)
        input = paddle.to_tensor(data=paddle.zeros(input_img.size()).requires_grad_(False), dtype=input_img.dtype)
        output = input + paddle.multiply(x=input_img, y=positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensor
        grad_input = None

        positive_mask_1 = paddle.to_tensor(data=(input_img > 0), dtype=grad_output.dtype)
        positive_mask_2 = paddle.to_tensor(data=(grad_output > 0), dtype=grad_output.dtype)

        input1 = paddle.to_tensor(data=paddle.zeros(input_img.size()).requires_grad_(False), dtype=input_img.dtype)
        output1 = input1 + paddle.multiply(x=grad_output, y=positive_mask_1)

        input = paddle.to_tensor(data=paddle.zeros(input_img.size()).requires_grad_(False),
                             dtype=input_img.dtype)
        grad_input = input + paddle.multiply(x=output1, y=positive_mask_2)

        return grad_input


class GuidedBackpropReLUasModule(paddle.nn.Layer):

    def __init__(self):
        super(GuidedBackpropReLUasModule, self).__init__()

    def forward(self, input_img):
        return GuidedBackpropReLU.apply(input_img)


class GuidedBackpropReLUModel:

    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = self.model.cuda()

    def forward(self, input_img):
        return self.model(input_img)

    def recursive_replace_relu_with_guidedrelu(self, module_top):
        for idx, module in module_top._sub_layers.items():
            self.recursive_replace_relu_with_guidedrelu(module)
            if module.__class__.__name__ == 'ReLU':
                #module_top._modules[idx] = GuidedBackpropReLU.apply
                module_top._sub_layers[idx] = GuidedBackpropReLU.apply
        print('b')

    def recursive_replace_guidedrelu_with_relu(self, module_top):
        try:
            for idx, module in module_top._sub_layers.items():
                self.recursive_replace_guidedrelu_with_relu(module)
                if module == GuidedBackpropReLU.apply:
                    module_top._sub_layers[idx] = paddle.nn.ReLU()
        except BaseException:
            pass

    def __call__(self, input_img, target_category=None):
        replace_all_layer_type_recursive(self.model,
                                         paddle.nn.ReLU,
                                         GuidedBackpropReLUasModule())

        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        loss = output[0, int(target_category)]

        #loss = paddle.to_tensor(data=loss, dtype=paddle.float32, stop_gradient=)
        #mask = paddle.ones_like(output, dtype='float32')
        #loss = paddle.fluid.layers.matmul(output, mask, transpose_y=True)
        #loss.clear_gradients()

        loss.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]
        output = output.transpose((1, 2, 0))

        replace_all_layer_type_recursive(self.model,
                                         GuidedBackpropReLUasModule,
                                         paddle.nn.ReLU())
        return output