import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pandas as pd


def concat_images(imga, imgb, orientation):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    # concatenate horizontally
    if orientation == 'horizontal':
        max_height = np.max([ha, hb])
        total_width = wa+wb
        new_img = np.zeros(shape=(max_height, total_width, 3))
        new_img[:ha, :wa] = imga
        new_img[:hb, wa:wa+wb] = imgb
    # concatenate vertically
    else:
        max_width = np.max([wa, wb])
        total_height = ha+hb
        new_img = np.zeros(shape=(total_height, max_width, 3))
        new_img[:ha, :wa] = imga
        new_img[ha:ha+hb, max_width-wb:] = imgb

    return new_img


def concat_n_images(image_path_list, orientation):
    """
    Combines N color images from a list of image paths.
    """
    output = None
    img_list = []
    for index, img_path in enumerate(image_path_list):
        img = plt.imread(img_path)[:, :, :3]
        if index < 2:
            img_c1 = img[0:180, :]
            img_list.append(img_c1)
            img_c2 = img[320:, :]
            img_list.append(img_c2)
        else:
            img_list.append(img)
    for i in range(len(img_list)):
        img = img_list[i]
        if i == 0:
            output = img
        else:
            output = concat_images(output, img, orientation)
    return output


def make_movie(image_folder, video_name):

    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    codec = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    # Original_path = r'C:\Users\Public\IPR\After_CSE\Sequence_Dev\20190320_103444_click_lefttop_570108590955\images\Raw_dR=10'
    CSE_path = r'C:\Users\Public\IPR\After_CSE\ROI1420TH+DL1.5+DLTH1.5'
    VisRes_path = r'C:\Users\Public\IPR\VisExps\VisRes\ROI1420TH+DL1.5+DLTH1.5'
    combined_path = r'C:\Users\Public\IPR\VisExps\Combined\ROI1420TH+DL1.5+DLTH1.5'
    orientation = 'vertical'            # or 'horizontal', don't have to change, double check with Rui

    # ==================================================================================================================
    CSE_path_list = []
    VisRes_path_list = []
    combined_path_list = []
    for folder in sorted(os.listdir(CSE_path)):
        CSE_path_list.append(os.path.join(CSE_path, folder, 'images\Raw_dR=10'))
    for folder in sorted(os.listdir(VisRes_path)):
        VisRes_path_list.append(os.path.join(VisRes_path, folder))
        combined_path_list.append(os.path.join(combined_path, folder))

    for i in range(len(CSE_path_list)):
        Original_path = CSE_path_list[i]
        VisRes = VisRes_path_list[i]
        combined = combined_path_list[i]
        video_name = os.path.join(combined, 'video.avi')

        print(Original_path)
        print(VisRes)
        print(combined)
        print(video_name)

        img_orig, img_vis = [], []
        for file in os.listdir(Original_path):
            img_orig.append(os.path.join(Original_path, file))

        for file in os.listdir(VisRes):
            img_vis.append(os.path.join(VisRes, file))

        if not os.path.exists(combined):
            os.makedirs(combined)

        col_val = ['img_pc', 'img_v', 'img_o']
        df = pd.DataFrame(columns=col_val)
        index = 0
        for img_v in img_vis:
            for img_o in img_orig:
                img_num = img_v.split('\\')[-1][10:16]
                if img_num in img_o.split('\\')[-1] and 'c1' in img_o and '_nc' in img_v:
                    img_pc = img_v.replace('_nc', '_pc')
                    images = [img_pc, img_v, img_o]
                    output = concat_n_images(images, orientation)
                    plt.imsave(os.path.join(combined, img_v.split('\\')[-1]), output)
                    plt.close()
                    print('Processed: {}/{}'.format(index, len(img_vis)/2))
                    df.loc[index, 'img_pc'] = img_pc
                    df.loc[index, 'img_v'] = img_v
                    df.loc[index, 'img_o'] = img_o
                    index += 1
        df.to_csv(os.path.join(combined, 'matched_pairs.csv'), header=True, index=False)
        make_movie(combined, video_name)