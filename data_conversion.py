
import os
import sys
import torch
import numpy as np
import matplotlib.pylab as plt
from datasets.data_process import load_scan, get_pixels_hu, resample, resample_both, plot_3d
import SimpleITK as sitk
import nibabel as nib
import h5py

# root = '../dataset/medical_image/'

root = f"../dataset/{sys.argv[1]}"

def convert_data(root):
    total_pids = 0
    raw_path = os.path.join(root, 'raw_data')
    img_path = os.path.join(raw_path, 'ct')
    pids = os.listdir(img_path)

    for pid in pids:

        if pid[0] == '.':
            continue

        total_pids += 1
        dates = os.listdir(os.path.join(raw_path, 'ct', pid))
        dates = [date for date in dates if date[0]!='.']
        date = dates[0] if dates[0] < dates[1] else dates[1]

        img_folder = os.path.join(raw_path, 'ct', pid, date)
        seg_file = os.path.join(raw_path, 'label', pid, '{}.nii.gz'.format(date))

        # ===== ct image and seg_mask
        scan = load_scan(img_folder)
        image = get_pixels_hu(scan)

        nii = nib.load(seg_file)
        nii_data = nii.get_fdata()
        seg_mask = torch.tensor(nii_data).permute(2, 1, 0)
        seg_mask = seg_mask.numpy().astype(np.int16)
        assert seg_mask.shape == image.shape, "Different dimension between CT image and segmentation mask"

        # ===== resample image and label
        image, seg_mask = resample_both(image, seg_mask, scan, new_spacing=[1.0, 1.0, 1.0])

        processed_path = os.path.join(root, 'processed_data')
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)
            print('create folder {}'.format(processed_path))
        out_path = os.path.join(processed_path, f'{pid}.hdf5')
        with h5py.File(out_path, 'w') as f:
            grp = f.create_group('ct')
            grp.create_dataset('image', data=image)
            grp.create_dataset('seg', data=seg_mask)
            print('export pid {} with ct image shape {}'.format(pid, image.shape))
    print("===> has finished processing all {} patients".format(total_pids))


# ====== plot
def plot_ct(idx, image, seg_mask):
    idx=150
    f, axarr = plt.subplots(1,3,figsize=(15,15))
    axarr[0].imshow(np.squeeze(image[idx, :, :]), cmap='gray',origin='lower');
    axarr[0].set_ylabel('Axial View',fontsize=14)
    axarr[0].set_xticks([])
    axarr[0].set_yticks([])
    axarr[0].set_title('CT',fontsize=14)

    axarr[1].imshow(np.squeeze(seg_mask[idx, :, :]), cmap='jet',origin='lower');
    axarr[1].axis('off')
    axarr[1].set_title('Mask',fontsize=14)

    axarr[2].imshow(np.squeeze(image[idx, :, :]), cmap='gray',alpha=1,origin='lower');
    axarr[2].imshow(np.squeeze(seg_mask[idx, :, :]),cmap='jet',alpha=0.5,origin='lower');
    axarr[2].axis('off')
    axarr[2].set_title('Overlay',fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)


if __name__ == "__main__":
    convert_data(root)



