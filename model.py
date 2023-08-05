


class SmiUnet(torch.nn.Module):
    def __init__(self):

        super(SmiUnet, self).__init__()

        self.model = UNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )


    def origin_forward(self,img):

        return torch.sigmoid(self.model(img))

    def get_biopsy(self,pro,grid ):

        smplers = F.grid_sample(pro, grid.unsqueeze(-2) * 2 - 1, align_corners=True).squeeze(-1)


        assert smplers.dim() == 4
        biopsy_all=F.adaptive_max_pool1d(smplers.squeeze(1),output_size=1).squeeze(-1)

        return biopsy_all

    def forward(self,img, grid):

        pro=self.model(img)

        smplers=F.grid_sample(pro, grid.unsqueeze(-2)*2-1 ,align_corners=True).squeeze(-1)
        assert smplers.dim()==4
        biopsy_all=F.adaptive_max_pool1d(smplers.squeeze(1),output_size=1).squeeze(-1)


        return  biopsy_all







