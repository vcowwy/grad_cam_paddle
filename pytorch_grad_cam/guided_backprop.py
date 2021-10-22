import numpy as np
import paddle
#from torch.autograd import Function
from pytorch_grad_cam.utils.find_layers import replace_all_layer_type_recursive


class GuidedBackpropReLU(paddle.autograd.PyLayer):

    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = paddle.addmm(
            paddle.zeros(input_img.size()).requires_grad_(False).type_as(
                input_img),
            input_img,
            positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None
        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = paddle.addmm(
            paddle.zeros(input_img.size()).requires_grad_(False).type_as(
                input_img),
            paddle.addmm(
                paddle.zeros(input_img.size()).requires_grad_(False).type_as(
                    input_img),
                grad_output,
                positive_mask_1),
            positive_mask_2)
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
        for idx, module in module_top._modules.items():
            self.recursive_replace_relu_with_guidedrelu(module)
            if module.__class__.__name__ == 'ReLU':
                module_top._modules[idx] = GuidedBackpropReLU.apply
        print('b')

    def recursive_replace_guidedrelu_with_relu(self, module_top):
        try:
            for idx, module in module_top._modules.items():
                self.recursive_replace_guidedrelu_with_relu(module)
                if module == GuidedBackpropReLU.apply:
                    module_top._modules[idx] = paddle.nn.ReLU()
        except BaseException:
            pass

    def __call__(self, input_img, target_category=None):
        replace_all_layer_type_recursive(self.model, paddle.nn.ReLU,
            GuidedBackpropReLUasModule())
        if self.cuda:
            input_img = input_img.cuda()
        input_img = input_img.requires_grad_(True)
        output = self.forward(input_img)
        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())
        loss = output[0, target_category]
        loss.backward(retain_graph=True)
        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]
        output = output.transpose((1, 2, 0))
        replace_all_layer_type_recursive(self.model,
            GuidedBackpropReLUasModule, paddle.nn.ReLU())
        return output
