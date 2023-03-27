# Autoencoder code

class Encoder(nn.Module):
    def __init__(self, num_channels, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(4608, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, num_channels):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 4608)
        self.conv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.Conv2d(256, num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(x.size(0), 512, 3, 3)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x


class AutoEncoder(nn.Module):
    def __init__(self, num_channels, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(num_channels, latent_dim)
        self.decoder = Decoder(latent_dim, num_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x