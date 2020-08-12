import typing

import torch
import torch.nn
import torch.utils.model_zoo

import thelper.nn
import thelper.nn.coordconv


def get_activation_layer(name: typing.AnyStr, *args, **kwargs) -> torch.nn.Module:
    # todo: support more prebuilt/custom layer types here, if needed...
    assert name in ["relu", "leaky_relu"]
    if name == "relu":
        return torch.nn.ReLU(inplace=True)
    elif name == "leaky_relu":
        return torch.nn.LeakyReLU(*args, inplace=True, **kwargs)


def get_norm_layer(name: typing.AnyStr, *args, **kwargs) -> torch.nn.Module:
    # todo: support more prebuilt/custom layer types here, if needed...
    assert name in ["batch", "layer"]
    if name == "batch":
        return torch.nn.BatchNorm2d(*args, **kwargs)
    elif name == "layer":
        return torch.nn.LayerNorm(*args, **kwargs)


class Module(torch.nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: typing.Optional[torch.nn.Module] = None,
            coordconv: bool = False,
            radius_channel: bool = True,
    ):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.downsample = downsample
        self.coordconv = coordconv
        self.radius_channel = radius_channel

    def _make_conv2d(self, *args, **kwargs):
        if self.coordconv:
            return thelper.nn.coordconv.CoordConv2d(*args, radius_channel=self.radius_channel, **kwargs)
        else:
            return torch.nn.Conv2d(*args, **kwargs)


class BasicBlock(Module):

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: typing.Optional[torch.nn.Module] = None,
            coordconv: bool = False,
            radius_channel: bool = True,
            activation: typing.AnyStr = "relu",
            norm: typing.AnyStr = "batch",
    ):
        super().__init__(inplanes, planes, stride, downsample, coordconv, radius_channel)
        self.conv1 = self._make_conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = get_norm_layer(norm, planes)
        self.activ = get_activation_layer(activation)
        self.conv2 = self._make_conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = get_norm_layer(norm, planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activ(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activ(out)
        return out


class Bottleneck(Module):
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: typing.Optional[torch.nn.Module] = None,
            coordconv: bool = False,
            radius_channel: bool = True,
            activation: typing.AnyStr = "relu",
            norm: typing.AnyStr = "batch",
    ):
        super().__init__(inplanes, planes, stride, downsample, coordconv, radius_channel)
        self.conv1 = self._make_conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.norm1 = get_norm_layer(norm, planes)
        self.conv2 = self._make_conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = get_norm_layer(norm, planes)
        self.conv3 = self._make_conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.norm3 = get_norm_layer(norm, planes * self.expansion)
        self.activ = get_activation_layer(activation)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activ(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activ(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activ(out)
        return out


class SqueezeExcitationLayer(torch.nn.Module):

    def __init__(
            self,
            channel: int,
            reduction: int = 16,
            activation: typing.AnyStr = "relu",
    ):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction),
            get_activation_layer(activation),
            torch.nn.Linear(channel // reduction, channel),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SqueezeExcitationBlock(Module):

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: typing.Optional[torch.nn.Module] = None,
            reduction: int = 16,
            coordconv: bool = False,
            radius_channel: bool = True,
            activation: typing.AnyStr = "relu",
            norm: typing.AnyStr = "batch",
    ):
        super().__init__(inplanes, planes, stride, downsample, coordconv, radius_channel)
        self.conv1 = self._make_conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = get_norm_layer(norm, planes)
        self.activ = get_activation_layer(activation)
        self.conv2 = self._make_conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = get_norm_layer(norm, planes)
        self.se = SqueezeExcitationLayer(planes, reduction, activation=activation)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activ(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activ(out)
        return out


class ResNet(thelper.nn.Module):

    def __init__(
            self,
            task: thelper.tasks.Task,
            block: typing.AnyStr = "thelper.nn.resnet.BasicBlock",
            layers: typing.Sequence[int] = [3, 4, 6, 3],
            strides: typing.Sequence[int] = [1, 2, 2, 2],
            input_channels: int = 3,
            flexible_input_res: bool = False,
            inplanes: int = 64,
            pool_size: int = 7,
            head_type: typing.Optional[typing.AnyStr] = None,
            coordconv: bool = False,
            radius_channel: bool = True,
            activation: typing.AnyStr = "relu",
            norm: typing.AnyStr = "batch",
            skip_max_pool: bool = False,
            pretrained: bool = False,
            conv1_config: typing.Sequence[int] = [7, 2, 3],
    ):
        # note: must always forward args to base class to keep backup
        super().__init__(task, **{k: v for k, v in vars().items() if k not in ["self", "task", "__class__"]})
        if isinstance(block, str):
            block = thelper.utils.import_class(block)
        assert issubclass(block, Module), "block type must be subclass of thelper.nn.resnet.Module"
        if isinstance(layers, str):
            assert layers in ["18", "34"], "unknown basic block layer depth string postfix"
            if layers == "18":
                layers = [2, 2, 2, 2]
            elif layers == "34":
                layers = [3, 4, 6, 3]
        assert isinstance(layers, list) and isinstance(strides, list), \
            "expected layers/strides to be provided as list of ints"
        assert len(layers) == len(strides), "layer/strides length mismatch"
        assert 0 < inplanes, "invalid inplanes count"
        # NOTE: conv1_config=[7,2,3] is the basic configuration of ResNet.
        #       other configuration more suitables for CIFAR for example can use conv1_config[3,1,1]
        assert isinstance(conv1_config, list) and \
            len(conv1_config) == 3 and all(isinstance(c, int) for c in conv1_config), \
            "conv1 configuration must be a list of 3 parameters defining [kernel_size,stride,padding]"
        self.input_channels = input_channels
        self.flexible_input_res = flexible_input_res
        self.pool_size = pool_size
        self.head_type = head_type
        self.coordconv = coordconv
        self.radius_channel = radius_channel
        self.pretrained = pretrained
        self.inplanes = inplanes
        self.conv1 = self._make_conv2d(
            in_channels=input_channels, out_channels=self.inplanes,
            kernel_size=conv1_config[0], stride=conv1_config[1],
            padding=conv1_config[2], bias=False)
        self.norm1 = get_norm_layer(norm, self.inplanes)
        self.activ = get_activation_layer(activation)
        if skip_max_pool:
            self.maxpool = None
        else:
            self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0], stride=strides[0], activation=activation)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=strides[1], activation=activation)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=strides[2], activation=activation)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=strides[3], activation=activation)
        self.out_features = inplanes * 8
        self.layer5 = None
        if len(layers) > 4:
            self.layer5 = self._make_layer(block, inplanes * 16, layers[4], stride=strides[4], activation=activation)
            self.out_features = inplanes * 16
        if flexible_input_res:
            self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            if pool_size < 1:
                raise AssertionError("invalid avg pool size for non-flex resolution")
            self.avgpool = torch.nn.AvgPool2d(pool_size, stride=1)
        self.out_features *= block.expansion
        self.fc = torch.nn.Linear(self.out_features, 1000)  # output type/count will be specialized by task after init
        self._init_weights(activation)
        if pretrained:
            # note: if using a non-default setup in the constructor, loading the pre-trained weights will most
            # likely fail as the weights are downloaded from the pytorch model zoo for the regular resnet impls
            import torchvision
            default_weights_mapping = {
                str([2, 2, 2, 2]) + str("BasicBlock"): "resnet18",
                str([3, 4, 6, 3]) + str("BasicBlock"): "resnet34",
                str([3, 4, 6, 3]) + str("Bottleneck"): "resnet50",
                str([3, 4, 23, 3]) + str("Bottleneck"): "resnet101",
                str([3, 8, 36, 3]) + str("Bottleneck"): "resnet152"
            }
            tag = str(layers) + block.__name__
            assert tag in default_weights_mapping, "could not find corresponding weight url"
            weights_url = torchvision.models.resnet.model_urls[default_weights_mapping[tag]]
            state_dict = torchvision.models.utils.load_state_dict_from_url(weights_url)
            self.load_state_dict(state_dict)
        if isinstance(task, thelper.tasks.Segmentation):
            # if base task is already associated with segmentation, add head attribute
            if head_type is not None:  # can also be manually defined (e.g. in autoencoder)
                assert isinstance(head_type, str) and head_type in ["fcn", "deeplabv3"], \
                    f"unrecognized head type ('{head_type}') for segmentation resnet"
            # note: head below will be fully instantiated when the task is assigned
            self.fc = None
        if task is not None:
            self.set_task(task)

    def _init_weights(self, activation):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, thelper.nn.coordconv.CoordConv2d) or \
                    isinstance(m, thelper.nn.coordconv.CoordConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.conv.weight, mode='fan_out', nonlinearity=activation)
            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, Bottleneck):
                torch.nn.init.constant_(m.norm3.weight, 0)
            elif isinstance(m, BasicBlock) or isinstance(m, SqueezeExcitationBlock):
                torch.nn.init.constant_(m.norm2.weight, 0)

    def _make_conv2d(self, *args, **kwargs):
        if self.coordconv:
            return thelper.nn.coordconv.CoordConv2d(*args, radius_channel=self.radius_channel, **kwargs)
        else:
            return torch.nn.Conv2d(*args, **kwargs)

    def _make_layer(self, block, planes, blocks, stride, activation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                self._make_conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes=planes, stride=stride,
                        downsample=downsample, activation=activation)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return torch.nn.Sequential(*layers)

    def get_embedding(self, x, pool=True):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activ(x)
        if self.maxpool is not None:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.layer5 is not None:
            x = self.layer5(x)
        if pool:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        if isinstance(self.task, thelper.tasks.Classification):
            return self.fc(self.get_embedding(x, pool=True))
        elif isinstance(self.task, thelper.tasks.Segmentation):
            return self.fc(self.get_embedding(x), pool=False)

    def set_task(self, task):
        assert isinstance(task, (thelper.tasks.Classification, thelper.tasks.Segmentation)), \
            "missing impl for non-classif task type"
        num_classes = len(task.class_names)
        if isinstance(task, thelper.tasks.Classification):
            if self.fc.out_features != num_classes:
                self.fc = torch.nn.Linear(self.out_features, num_classes)
        elif isinstance(task, thelper.tasks.Segmentation):
            import torchvision.models.segmentation
            # note: heads below will be fully reinstantiated when the output class count changes
            if self.fc is None or self.fc[len(self.fc) - 1].out_channels != num_classes:
                if self.head_type == "fcn":
                    self.fc = torchvision.models.segmentation.fcn.FCNHead(self.out_features, num_classes)
                elif self.head_type == "deeplabv3":
                    self.fc = torchvision.models.segmentation.deeplabv3.DeepLabHead(self.out_features, num_classes)
        self.task = task


class ConvTailNet(torch.nn.Module):
    """DEPRECATED. Will be removed in a future version."""

    def __init__(self, n_inputs, num_classes):
        super(ConvTailNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(n_inputs, n_inputs, kernel_size=1, bias=False)
        self.relu = torch.nn.ReLU(True)
        self.conv2 = torch.nn.Conv2d(n_inputs, n_inputs, kernel_size=1, bias=False)
        self.conv3 = torch.nn.Conv2d(n_inputs, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = torch.add(x0, x)
        x = self.conv3(x)
        return x


class ResNetFullyConv(ResNet):
    """DEPRECATED. Will be removed in a future version. Use the torchvision segmentation models or the ResNet above instead."""

    def __init__(self, task, block="thelper.nn.resnet.BasicBlock",
                 layers=[3, 4, 6, 3], strides=[1, 2, 2, 2], input_channels=3,
                 flexible_input_res=False, pool_size=7, coordconv=False,
                 radius_channel=True, pretrained=False):
        super().__init__(task=task, block=block, layers=layers, strides=strides, input_channels=input_channels,
                         flexible_input_res=flexible_input_res, pool_size=pool_size, coordconv=coordconv,
                         radius_channel=radius_channel, pretrained=pretrained)
        self.set_task(task)

    def forward(self, x):
        x = self.get_embedding(x, pool=False)
        x = self.avgpool(x)
        x = self.fc(x)
        x = torch.squeeze(x)
        return x

    def set_task(self, task):
        assert isinstance(task, thelper.tasks.Classification), "missing impl for non-classif task type"
        num_classes = len(task.class_names)
        self.fc = ConvTailNet(self.out_features, num_classes)
        self.task = task


class FCResNet(ResNet):
    """Fully Convolutional ResNet converter for pre-trained classification models."""

    def __init__(self, task, ckptdata, map_location="cpu", avgpool_size=0):
        if isinstance(ckptdata, str):
            ckptdata = thelper.utils.load_checkpoint(ckptdata, map_location=map_location)
        model_type = ckptdata["model_type"]
        if model_type != "thelper.nn.resnet.ResNet":
            raise AssertionError("cannot convert non-resnet model to fully conv with this impl")
        model_params = ckptdata["model_params"]
        if isinstance(ckptdata["task"], str):
            old_model_task = thelper.tasks.create_task(ckptdata["task"])
        else:
            old_model_task = ckptdata["task"]
        self.task = None
        self.avgpool_size = avgpool_size
        super().__init__(old_model_task, **model_params)
        self.load_state_dict(ckptdata["model"], strict=False)  # assumes model always stored as weight dict
        self.finallayer = torch.nn.Conv2d(self.out_features, self.fc.out_features, kernel_size=1)
        self.finallayer.weight = \
            torch.nn.Parameter(self.fc.weight.view(self.fc.out_features, self.out_features, 1, 1))
        self.finallayer.bias = torch.nn.Parameter(self.fc.bias)
        self.set_task(task)

    def forward(self, x):
        x = self.get_embedding(x, pool=False)
        if self.avgpool_size > 0:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=self.avgpool_size, stride=1)
        x = self.finallayer(x)
        return x

    def set_task(self, task):
        assert isinstance(task, (thelper.tasks.Segmentation, thelper.tasks.Classification)), \
            "missing impl for non-segm/classif task type"
        num_classes = len(task.class_names)
        if self.fc.out_features != num_classes:
            self.fc = torch.nn.Linear(self.out_features, num_classes)
            self.finallayer = torch.nn.Conv2d(self.out_features, num_classes, kernel_size=1)
            self.finallayer.weight = \
                torch.nn.Parameter(self.fc.weight.view(self.fc.out_features, self.out_features, 1, 1))
            self.finallayer.bias = torch.nn.Parameter(self.fc.bias)
        self.task = task


class AutoEncoderResNet(ResNet):
    """Autoencoder-classifier architecture based on ResNet blocks+layers configurations."""

    def __init__(self, task, output_pads=None, **kwargs):
        assert isinstance(task, thelper.tasks.Classification)
        super().__init__(task, activation="leaky_relu", **kwargs)
        convt = thelper.nn.coordconv.CoordConvTranspose2d if self.coordconv else torch.nn.ConvTranspose2d
        self.decoder_top = torch.nn.Sequential(
            thelper.nn.coordconv.CoordConv2d(
                self.out_features, self.out_features, kernel_size=1, stride=1, padding=0
            ),
            torch.nn.BatchNorm2d(self.out_features),
            torch.nn.LeakyReLU(),
        )
        self.decoder_depths = [self.out_features // 2 ** d for d in range(0, 5)]
        self.output_pads = output_pads if not None else [1, 1, 1, 1, 1]
        self.decoder_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                convt(depth, depth // 2, kernel_size=3, stride=2, padding=1, output_padding=out_pad),
                torch.nn.BatchNorm2d(depth // 2),
                torch.nn.LeakyReLU(),
            )
            for depth, out_pad in zip(self.decoder_depths, self.output_pads)
        ])
        self.decoder_bottom = torch.nn.Sequential(
            thelper.nn.coordconv.CoordConv2d(
                self.decoder_depths[-1] // 2, self.input_channels,
                kernel_size=3, stride=1, padding=1,
            ),
            torch.nn.Tanh()
        )
        self._init_weights(activation="leaky_relu")
        # note: cannot rely on pretrained imagenet weights since we reset just above

    def forward(self, input):
        featmap = self.get_embedding(input, pool=False)
        embedding = self.avgpool(featmap)
        embedding = embedding.view(embedding.size(0), -1)
        class_logits = self.fc(embedding)
        featmap = self.decoder_top(featmap)
        for decoder_layer in self.decoder_layers:
            featmap = decoder_layer(featmap)
        reconstruction = self.decoder_bottom(featmap)
        return class_logits, reconstruction


class FakeModule(torch.nn.Module):
    def forward(self, inputs):
        return inputs


class AutoEncoderSkipResNet(ResNet):
    """Autoencoder-U-Net architecture based on ResNet blocks+layers configurations."""

    def __init__(self, task, output_pads=None, decoder_dropout=False, dropout_prob=0.1, **kwargs):
        assert isinstance(task, thelper.tasks.Segmentation)
        super().__init__(task, activation="leaky_relu", **kwargs)
        convt = thelper.nn.coordconv.CoordConvTranspose2d if self.coordconv else torch.nn.ConvTranspose2d
        pass_through_layer = FakeModule()
        self.decoder_top = torch.nn.Sequential(
            thelper.nn.coordconv.CoordConv2d(
                self.out_features, self.out_features, kernel_size=1, stride=1, padding=0
            ),
            torch.nn.BatchNorm2d(self.out_features),
            torch.nn.Dropout2d(p=dropout_prob) if decoder_dropout else pass_through_layer,
            torch.nn.LeakyReLU(),
        )
        self.output_pads = output_pads if not None else [1, 1, 1, 1, 1]
        self.ae_decoder_depths = [self.out_features // 2 ** d for d in range(0, 5)]
        self.ae_decoder_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                convt(depth, depth // 2, kernel_size=3, stride=2, padding=1, output_padding=out_pad),
                torch.nn.BatchNorm2d(depth // 2),
                torch.nn.Dropout2d(p=dropout_prob) if decoder_dropout else pass_through_layer,
                torch.nn.LeakyReLU(),  # try w/ regular? @@@@
            )
            for depth, out_pad in zip(self.ae_decoder_depths, self.output_pads)
        ])
        self.ae_decoder_bottom = torch.nn.Sequential(
            thelper.nn.coordconv.CoordConv2d(
                self.ae_decoder_depths[-1] // 2, self.input_channels,
                kernel_size=3, stride=1, padding=1,
            ),
            torch.nn.Tanh()
        )
        self.unet_decoder_depths = [
            (self.out_features, self.out_features // 2, self.out_features // 2),
            (self.out_features, self.out_features // 4, self.out_features // 4),
            (self.out_features // 2, self.out_features // 8, self.out_features // 8),
            (self.out_features // 4, self.out_features // 8, self.out_features // 8),
            (self.out_features // 4, self.out_features // 8, self.out_features // 8),
        ]
        self.unet_decoder_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                convt(d_in, d_mid, kernel_size=3, stride=2, padding=1, output_padding=out_pad),
                torch.nn.BatchNorm2d(d_mid),
                torch.nn.LeakyReLU(),  # try w/ regular? @@@@
                self._make_conv2d(d_mid, d_out, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(d_out),
                torch.nn.Dropout2d(p=dropout_prob) if decoder_dropout else pass_through_layer,
                torch.nn.LeakyReLU(),  # try w/ regular? @@@@
            )
            for (d_in, d_mid, d_out), out_pad in zip(self.unet_decoder_depths, self.output_pads)
        ])
        assert self.unet_decoder_depths[-1][2] > len(self.task.class_names), "woopsie"
        self.unet_decoder_bottom = torch.nn.Conv2d(
            self.unet_decoder_depths[-1][2], len(self.task.class_names),
            kernel_size=1, stride=1, padding=0,
        )
        self._init_weights(activation="leaky_relu")
        # note: cannot rely on pretrained imagenet weights since we reset just above

    def forward(self, input):
        # forward while keeping refs for latent build? @@@
        encoder1 = self.activ(self.norm1(self.conv1(input)))
        if self.maxpool is not None:
            encoder2 = self.layer1(self.maxpool(encoder1))
        else:
            encoder2 = self.layer1(encoder1)
        encoder3 = self.layer2(encoder2)
        encoder4 = self.layer3(encoder3)
        encoder5 = self.layer4(encoder4)
        assert self.layer5 is None, "missing e5+ impl"
        featmap = self.decoder_top(encoder5)
        reconstruction = featmap
        for decoder_layer in self.ae_decoder_layers:
            reconstruction = decoder_layer(reconstruction)
        reconstruction = self.ae_decoder_bottom(reconstruction)
        encoder_maps = [None, encoder4, encoder3, encoder2, encoder1]
        for decoder_layer, encoder_map in zip(self.unet_decoder_layers, encoder_maps):
            if encoder_map is not None:
                featmap = torch.cat([featmap, encoder_map], dim=1)
            featmap = decoder_layer(featmap)
        class_logits = self.unet_decoder_bottom(featmap)
        return class_logits, reconstruction
