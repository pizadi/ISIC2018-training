class UNetRes50(BaseModel):
    def __init__(self, learning_rate=None,  loss_fn=None, optimizer=None, device=None):
        super(UNetRes50, self).__init__()

        if (device is None):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if (learning_rate is None):
            self.learning_rate = 1e-4
        else:
            self.learning_rate = learning_rate

        resnet50 = torchvision.models.resnet50(weights="DEFAULT")
        self.e1 = nn.Sequential(resnet50._modules['conv1'], resnet50._modules['bn1'], resnet50._modules['relu'])
        self.e2 = nn.Sequential(resnet50._modules['maxpool'], resnet50._modules['layer1'])
        self.e3 = nn.Sequential(resnet50._modules['layer2'])
        self.e4 = nn.Sequential(resnet50._modules['layer3'])
        
        for param in e1.parameters():
            param.requires_grad_(False)
            
        for param in e2.parameters():
            param.requires_grad_(False)
            
        for param in e3.parameters():
            param.requires_grad_(False)
            
        for param in e4.parameters():
            param.requires_grad_(False)
        
        self.c1 = self.convblock(1024, 512)
        self.c2 = self.convblock(512, 256)
        self.c3 = self.convblock(192, 128)
        self.c4 = self.convblock(67, 64)
        
        self.u1 = nn.ConvTranspose2d(1024, 512, (2, 2), stride=(2, 2))
        self.u2 = nn.ConvTranspose2d(512, 256, (2, 2), stride=(2, 2))
        self.u3 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))
        self.u4 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
        
        self.out = nn.Conv2d(64, 1, (1, 1), padding="same")

        if (loss_fn is None):
            self.loss_fn = nn.BCELoss()
        else:
            self.loss_fn = loss_fn

        if (optimizer is None):
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optimizer(self.parameters(), lr=self.learning_rate)


        self = self.to(self.device)

    def convblock(self, in_features, out_features):
        block = nn.Sequential(
            nn.Conv2d(in_features, out_features, (3, 3), padding="same"),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, (3, 3), padding="same"),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
            )
        block = block.to(self.device)
        return block


    def forward(self, X):
        X = X.to(self.device)

        s1 = self.e1(X)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        
        s4 = self.u1(s4)
        s3 = self.c1(torch.cat([s3, s4], dim=1))
        s4 = None

        s3 = self.u2(s3)
        s2 = self.c2(torch.cat([s2, s3], dim=1))
        s3 = None

        s2 = self.u3(s2)
        s1 = self.c3(torch.cat([s1, s2], dim=1))
        s2 = None

        s1 = self.u4(s1)
        X = self.c4(torch.cat([X, s1], dim=1))
        s1 = None

        X = torch.sigmoid(self.out(X))

        return X
