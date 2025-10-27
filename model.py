import torch
import torch.nn as nn


# Backbone
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU() if activation else nn.Identity()
            
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BottleNeck(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, activation=True, shortcut=True):
        super().__init__()
        self.use_shortcut = shortcut
        self.conv1 = Conv(in_ch, out_ch, kernel, stride, padding, activation=activation)
        self.conv2 = Conv(out_ch, out_ch, kernel, stride, padding, activation=activation)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.use_shortcut and identity.shape == out.shape:
            out = out + identity  # Not in-place
        
        return out


class C2f(nn.Module):
    def __init__(self, in_ch, out_ch, num_bottlenecks, kernel=1, stride=1, padding=1):
        super().__init__()
        self.hidden = max(1, out_ch // 2)
        self.conv1 = Conv(in_ch, 2 * self.hidden, kernel, stride, padding)
        self.necks = nn.ModuleList([BottleNeck(self.hidden, self.hidden) for _ in range(max(0, int(num_bottlenecks)))])
        concat_in_ch = (len(self.necks) + 2) * self.hidden
        self.conv2 = Conv(concat_in_ch, out_ch, kernel, stride, padding)

    def forward(self, x):
        y = self.conv1(x)
        
        # Split along channel axis
        y1 = y[:, :self.hidden, :, :]
        y2 = y[:, self.hidden:self.hidden * 2, :, :]
        
        # Collect outputs
        outputs = [y1]
        out = y1
        for neck in self.necks:
            out = neck(out)
            outputs.append(out)
        outputs.append(y2)
        
        # Concatenate and process
        out_cat = torch.cat(outputs, dim=1)
        return self.conv2(out_cat)


class SPPF(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, stride=1, padding=None):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = kernel // 2 if padding is None else padding
        
        self.conv1 = Conv(in_ch, in_ch // 2, kernel=1, stride=1, padding=0)
        self.conv2 = Conv(in_ch // 2 * 4, out_ch, kernel=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(self.kernel, self.stride, self.padding)

    def forward(self, x):
        x = self.conv1(x)
        
        # Perform pooling in 3 stages
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        
        # Concatenate
        y = torch.cat([x, y1, y2, y3], dim=1)
        return self.conv2(y)


# Neck
class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.d, self.w, self.r = 1/3, 1/4, 2.0

        self.c2f_1 = C2f(int(1024 + 256), int(256), int(3*self.d)) 
        self.c2f_2 = C2f(int(256 + 128), int(128), int(3*self.d)) 
        self.c2f_3 = C2f(int(128 + 256), int(256), int(3*self.d))
        self.c2f_4 = C2f(int(256 + 1024), int(512), int(3*self.d))

        self.down_1 = nn.Sequential(
            Conv(int(128), int(128), kernel=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.down_2 = nn.Sequential(
            Conv(int(256), int(256), kernel=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x_res_1, x_res_2, x):
        # Save for later use
        res_1 = x
        
        # Top-down path
        p5 = nn.functional.interpolate(x, size=(x_res_2.shape[2], x_res_2.shape[3]), mode='nearest')
        p5 = torch.cat([p5, x_res_2], dim=1)
        p5 = self.c2f_1(p5)
        
        p4 = nn.functional.interpolate(p5, size=(x_res_1.shape[2], x_res_1.shape[3]), mode='nearest')
        p4 = torch.cat([p4, x_res_1], dim=1)
        out_1 = self.c2f_2(p4)
        
        # Bottom-up path
        n4 = self.down_1(out_1)
        n4 = nn.functional.interpolate(n4, size=(p5.shape[2], p5.shape[3]), mode='nearest')
        n4 = torch.cat([n4, p5], dim=1)
        out_2 = self.c2f_3(n4)
        
        n5 = self.down_2(out_2)
        n5 = nn.functional.interpolate(n5, size=(res_1.shape[2], res_1.shape[3]), mode='nearest')
        n5 = torch.cat([n5, res_1], dim=1)
        out_3 = self.c2f_4(n5)

        return out_1, out_2, out_3


# Head
class DFL(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(ch, 1, kernel_size=1, stride=1, bias=True)
        
        # Initialize with [0, ..., ch-1]
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        with torch.no_grad():
            self.conv.weight[:] = x.unsqueeze(0)

    def forward(self, x):
        # x: [bs, 4*ch, num_anchors]
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(1, 2)  # [bs, ch, 4, num_anchors]
        x = x.softmax(1)
        x = self.conv(x)  # [bs, 1, 4, num_anchors]
        return x.view(b, 4, a)


class Head(nn.Module):
    def __init__(self, ch=16, num_classes=80):
        super().__init__()
        self.ch = ch
        self.nc = num_classes
        self.coordinates = ch * 4
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        
        # Box prediction heads
        self.box = nn.ModuleList([
            nn.Sequential(
                Conv(128, self.coordinates, kernel=3, stride=1, padding=1),
                Conv(self.coordinates, self.coordinates, kernel=3, stride=1, padding=1),
                nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                Conv(256, self.coordinates, kernel=3, stride=1, padding=1),
                Conv(self.coordinates, self.coordinates, kernel=3, stride=1, padding=1),
                nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                Conv(512, self.coordinates, kernel=3, stride=1, padding=1),
                Conv(self.coordinates, self.coordinates, kernel=3, stride=1, padding=1),
                nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)
            ),
        ])
        
        # Class prediction heads
        self.cls = nn.ModuleList([
            nn.Sequential(
                Conv(128, self.nc, kernel=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                Conv(256, self.nc, kernel=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                Conv(512, self.nc, kernel=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)
            )
        ])

        self.dfl = DFL(ch)
    
    def forward(self, x):
        # Process each scale
        outputs = []
        for i in range(len(self.box)):
            box_pred = self.box[i](x[i])
            cls_pred = self.cls[i](x[i])
            outputs.append(torch.cat([box_pred, cls_pred], dim=1))

        if self.training:
            return outputs
        
        # Inference mode
        anchors, strides = (t.transpose(0, 1) for t in self.make_anchors(outputs, self.stride))
        
        x_cat = torch.cat([out.view(outputs[0].shape[0], self.coordinates + self.nc, -1) 
                          for out in outputs], dim=2)
        
        box, cls = x_cat.split([self.coordinates, self.nc], dim=1)
        
        dfl_box = self.dfl(box)
        lt, rb = dfl_box.chunk(2, 1)
        
        x1y1 = anchors.unsqueeze(0) - lt
        x2y2 = anchors.unsqueeze(0) + rb
        
        box = torch.cat([((x1y1 + x2y2) / 2), (x2y2 - x1y1)], dim=1)
        
        return torch.cat([box * strides, cls.sigmoid()], dim=1)
    
    def make_anchors(self, x, strides, offset=0.5):
        anchor_points, stride_tensor = [], []
        dtype, device = x[0].dtype, x[0].device

        for i, stride in enumerate(strides):
            _, _, h, w = x[i].shape
            sx = torch.arange(w, device=device, dtype=dtype) + offset
            sy = torch.arange(h, device=device, dtype=dtype) + offset
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack([sx, sy], -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        
        return torch.cat(anchor_points), torch.cat(stride_tensor)