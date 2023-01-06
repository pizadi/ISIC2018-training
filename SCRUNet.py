class SCRUNet(LossNormModel):
    def __init__(self, learning_rate=None,  loss_fn=None, optimizer=None, device=None, rep=None, pass_features=None):
        super(SCRUNet, self).__init__()

        if (device is None):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if (learning_rate is None):
            self.learning_rate = 1e-4
        else:
            self.learning_rate = learning_rate
        
        if (rep is None):
            self.rep = 5
        else:
            self.rep = rep
            
        if (pass_features is None):
            self.pass_features = 64
        else:
            self.pass_features = pass_features

        resnet50 = torchvision.models.resnet50(weights="DEFAULT")
        self.e1 = nn.Sequential(resnet50._modules['conv1'], resnet50._modules['bn1'], resnet50._modules['relu'])
        self.e2 = nn.Sequential(resnet50._modules['maxpool'], resnet50._modules['layer1'])
        self.e3 = nn.Sequential(resnet50._modules['layer2'])
        self.e4 = nn.Sequential(resnet50._modules['layer3'])
        
        for param in self.e1.parameters():
            param.requires_grad_(False)
            
        for param in self.e2.parameters():
            param.requires_grad_(False)
            
        for param in self.e3.parameters():
            param.requires_grad_(False)
            
        for param in self.e4.parameters():
            param.requires_grad_(False)
        
        self.c1 = self.convblock(1024+self.pass_features, 512)
        self.c2 = self.convblock(512+self.pass_features, 256)
        self.c3 = self.convblock(192+self.pass_features, 64)
        self.c4 = self.convblock(67+self.pass_features, 3)
        
        self.u1 = nn.ConvTranspose2d(1024, 512, (2, 2), stride=(2, 2))
        self.u2 = nn.ConvTranspose2d(512, 256, (2, 2), stride=(2, 2))
        self.u3 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))
        self.u4 = nn.ConvTranspose2d(64, 64, (2, 2), stride=(2, 2))
    
        self.p1 = self.convblock(512, self.pass_features)
        self.p2 = self.convblock(256, self.pass_features)
        self.p3 = self.convblock(64, self.pass_features)
        self.p4 = self.convblock(3, self.pass_features)
        
        self.out = nn.Conv2d(self.pass_features, 1, (1, 1), padding="same")

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
        
        b, _, h, w = X.shape
        h1 = torch.zeros((b, self.pass_features, h, w)).to(self.device)
        h2 = torch.zeros((b, self.pass_features, h//2, w//2)).to(self.device)
        h3 = torch.zeros((b, self.pass_features, h//4, w//4)).to(self.device)
        h4 = torch.zeros((b, self.pass_features, h//8, w//8)).to(self.device)

        for _ in range(self.rep):
            s3 = self.c1(torch.cat([s3, self.u1(s4), h4], dim=1))
            h4 = self.p1(s3)

            s2 = self.c2(torch.cat([s2, self.u2(s3), h3], dim=1))
            h3 = self.p2(s2)

            s1 = self.c3(torch.cat([s1, self.u3(s2), h2], dim=1))
            h2 = self.p3(s1)

            X = self.c4(torch.cat([X, self.u4(s1), h1], dim=1))
            h1 = self.p4(X)
            
        X = torch.sigmoid(self.out(h1))

        return X