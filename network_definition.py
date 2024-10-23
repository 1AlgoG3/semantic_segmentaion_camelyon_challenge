import inspect
import torch
import torch.nn as nn

class Net(nn.Module):
    """ A base class provides a common weight initialisation scheme."""

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            if "norm" in classname.lower():
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if "linear" in classname.lower():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return x
    

class DoubleConv(Net):
    
    def __init__(self, in_ch, out_ch, kernel_size=3, activation=nn.ReLU):
        super(DoubleConv, self).__init__()
        
        if 'inplace' in inspect.signature(activation).parameters:
            self.activation=activation(inplace=True)
        else:
            self.activation=activation()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size),
            self.activation,
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size),
            self.activation,
        )
        
        self.weights_init()
        
    def forward(self, x):
        return self.conv(x)

class UNet(Net):
    
    def __init__(self, input_channels, n_classes, activation, dropout_prob=0.1):
        super(UNet, self).__init__()
        
        self.dropout_prob=dropout_prob
        self.n_classes=n_classes
        
        self.max_pool_2x2=nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1x1=nn.Conv2d(64, n_classes, kernel_size=1)
        self.down_double_conv1=DoubleConv(in_ch=input_channels, out_ch=64, activation=activation)
        self.down_double_conv2=DoubleConv(in_ch=64, out_ch=128, activation=activation)
        self.down_double_conv3=DoubleConv(in_ch=128, out_ch=256, activation=activation)
        self.down_double_conv4=DoubleConv(in_ch=256,out_ch=512,activation=activation)
        self.down_double_conv5=DoubleConv(in_ch=512,out_ch=1024,activation=activation)
        
        self.dropout = nn.Dropout(p=self.dropout_prob)
        
        self.up_conv1=nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=2, stride=2)
        self.up_conv2=nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=2, stride=2)
        self.up_conv3=nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2, stride=2)
        self.up_conv4=nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2, stride=2)

        self.up_double_conv1=DoubleConv(in_ch=1024, out_ch=512, activation=activation)
        self.up_double_conv2=DoubleConv(in_ch=512, out_ch=256, activation=activation)
        self.up_double_conv3=DoubleConv(in_ch=256, out_ch=128, activation=activation)
        self.up_double_conv4=DoubleConv(in_ch=128, out_ch=64, activation=activation)
        
        self.skip_coords=[]
        
        self.weights_init()
          
    def forward(self, patch):
        x1=self.down_double_conv1(patch)
        x1 = self.dropout(x1) 
        x2=self.max_pool_2x2(x1)
        
        x3=self.down_double_conv2(x2)
        x3 = self.dropout(x3) 
        x4=self.max_pool_2x2(x3)
        
        x5=self.down_double_conv3(x4)
        x6=self.max_pool_2x2(x5)
        
        x7=self.down_double_conv4(x6)
        x8=self.max_pool_2x2(x7)
        
        x9=self.down_double_conv5(x8)

        x10=self.up_conv1(x9)
        skip1, skip_cords1=self.get_skip_tensor(x7, x10)
        self.skip_coords.append(skip_cords1)
        concat1=torch.concat([skip1, x10], axis=1)
        x11=self.up_double_conv1(concat1)
        
        x12=self.up_conv2(x11)
        skip2, skip_cords2=self.get_skip_tensor(x5, x12)
        self.skip_coords.append(skip_cords2)
        concat2=torch.concat([skip2, x12], axis=1)
        x13=self.up_double_conv2(concat2)

        x14=self.up_conv3(x13)
        skip3, skip_cords3=self.get_skip_tensor(x3, x14)
        self.skip_coords.append(skip_cords3)
        concat3=torch.concat([skip3, x14], axis=1)
        x15=self.up_double_conv3(concat3)

        x16=self.up_conv4(x15)
        skip4, skip_cords4=self.get_skip_tensor(x1,x16)
        self.skip_coords.append(skip_cords4)
        concat4=torch.concat([skip4,x16], axis=1)
        x17=self.up_double_conv4(concat4)
    
        x18=self.conv_1x1(x17)
        
        return x18
    
    @staticmethod
    def get_skip_tensor(source_tensor, target_tensor):
        assert target_tensor.shape[:2]==source_tensor.shape[:2], "Batch and channel dimensions do not match"
        source_shape=source_tensor.shape[2]
        target_shape=target_tensor.shape[2]

        diff=(source_shape-target_shape)/2

        if diff%2==0:
            diff=round((source_shape-target_shape)/2)
            delta=0
            skip_tensor=source_tensor[:,:,diff:source_shape-diff,diff:source_shape-diff]
        else:
            diff=round(diff)
            delta=target_shape-(source_shape-2*diff)
            skip_tensor=source_tensor[:,:,diff:source_shape-diff+delta,diff:source_shape-diff+delta]

        assert skip_tensor.shape==target_tensor.shape, f"Shape mismatch: {skip_tensor.shape} vs {target_tensor.shape}"

        return skip_tensor, f":,:,{diff}:{source_shape}-{diff}+{delta},{diff}:{source_shape}-{diff}+{delta}"