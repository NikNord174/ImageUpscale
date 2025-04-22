from src.data.up_dataset import UpDataset


if __name__ == '__main__':

    im = '/workdir/data/poly-Ni_scan6_15kV_100pA_WD8-1_bin1_sat0100_exp140ms_gain00_gut00_step3500nm_2303points_.up2'

    dataset = UpDataset(file_path=im)

    sample = dataset[0]
    print(sample[0].shape, sample[1].shape)
