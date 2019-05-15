#######################################################################
#       This file is for generating ROI_imgs with correct file path   #
#######################################################################

import pandas as pd
import numpy as np
import os
import cv2


"""
    How to use the file:
        (1): update the first three file paths
        (2): choose the method to select region ( Now only use coordinate to classify region instead of 'region' in file name)
        (3): update dR_list, usually don't have to change 
    The file path generated will be: 
        person\Detection_Type\ region\hand\z
    
"""


def main():

    # sequence_Data = False
    # CSE_BaseFolder = r'C:\Users\Public\IPR\After_CSE\20190410_225624\TF_classification_focls-20-2.655'
    # lists = sorted(os.listdir(CSE_BaseFolder))
    # lists = ['20190423_132058_20190411_154558_CYCLE1_OFFICIAL',
    #          '20190423_132413_20190411_164820_C2_left_5Regions',
    #          '20190423_132454_20190411_165531_C2_5Regions',
    #          ]
    # only the images in CSE outputs will be used to crop ROI
    # original_imgs = [r'C:\Users\Public\IPR\DataFromServer\Data\fromEdge\DL_Datasets\Dataset_CSE_E2E_C1_C2',
    #                  r'C:\Users\Public\IPR\DataFromServer\Data\fromEdge\DL_Datasets\Dataset_CSE_E2E_C2',
    #                  r'C:\Users\Public\IPR\DataFromServer\Data\fromEdge\DL_Datasets\Dataset_CSE_E2E_C2',
    #                  ]
    # !!!!!!!!!!!!!!!!!!Check if the output folder is already exist before running the code!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # path_ROI = r'C:\Users\Public\IPR\VisExps\ROI_imgs\20190410_225624\TF_classification_focls-20-2.655'
    # The pose_name information must be the same as the info originally specified in the E2E evaluation
    # pose_name = '225624_20'
    # sequence_Data = False
    # generate_OTHERS = True
    # dR_list = [10]      # Most times dR only = 10

    # sequence_Data = True:
    CSE_BaseFolder = r'C:\Users\Public\IPR\After_CSE\ROI1420TH+DL1.5+DLTH1.5'
    # for using Sorted lists: need to make sure the order of the datasets of CSE_BaseFolder should be the same as datasets under original_imgs
    # lists = sorted(os.listdir(CSE_BaseFolder))
    # original_imgs = sorted(os.listdir(r'C:\Users\Public\IPR\DataFromServer\Data\fromEdge\For_Example'))

    lists = ['20190409_141024_click_leftbottom_570377438047',
             '20190409_141749_doubleclick_leftbottom_570377646410'
             ]
    original_imgs = [r'C:\Users\Public\IPR\DataFromServer\Data\fromEdge\20190128_Susan_Sequence_Data\click_leftbottom_570377438047',
                     r'C:\Users\Public\IPR\DataFromServer\Data\fromEdge\20190128_Susan_Sequence_Data\doubleclick_leftbottom_570377646410'
                     ]
    path_ROI = r'C:\Users\Public\IPR\VisExps\ROI_imgs\ROI1420TH+DL1.5+DLTH1.5'
    pose_name = 'ROI1484TH+DL1.5+DLTH1.5'
    sequence_Data = True
    generate_OTHERS = True
    dR_list = [10]  # Most times dR only = 10

    # ========================
    if sequence_Data:
        detect_list = ['Detect_TP', 'Detect_FP']
    else:
        detect_list = ['Detect_TP']

    for list_index in range(len(lists)):

        Curr_save = os.path.join(path_ROI, lists[list_index])
        if not os.path.isdir(Curr_save):
            os.makedirs(Curr_save)

        for dr in range(len(dR_list)):
            for detect in detect_list:
                Curr_folder = os.path.join(CSE_BaseFolder, lists[list_index], 'overall_separated_dR={}\{}'.format(dR_list[dr], detect))
                print(Curr_folder)
                for entry in os.listdir(Curr_folder):
                    selectedPath = os.path.join(Curr_folder, entry, 'selected.csv')
                    print(selectedPath)
                    selected_df = pd.read_csv(selectedPath)
                    for i in range(len(selected_df)):
                        person = selected_df.TesterName[i].strip(' ')
                        picture_name = selected_df.PictureName[i]
                        Detection_Type = selected_df.Detection_Type_Index[i]
                        if Detection_Type == 6:
                            Detection_Type = 'MissClassifi_H2T'
                        elif Detection_Type == 7:
                            Detection_Type = 'MissClassifi_T2H'
                        elif Detection_Type == 8:
                            Detection_Type = 'TP_Hover'
                        elif Detection_Type == 9:
                            Detection_Type = 'TP_Touch'

                        Pose_X = selected_df.loc[i, '(1) PoseX (Matlab frame) - {}'.format(pose_name)]
                        Pose_Y = selected_df.loc[i, '(1) PoseY (Matlab frame) - {}'.format(pose_name)]

                        if not sequence_Data:
                            if 11 < Pose_X <= 266 and 11 < Pose_Y <= 350:
                                region = 'top_left'
                            elif 11 < Pose_X <= 266 and 451 < Pose_Y <= 790:
                                region = 'bottom_left'
                            elif 1015 < Pose_X <= 1270 and 11 < Pose_Y <= 350:
                                region = 'top_right'
                            elif 1015 < Pose_X <= 1270 and 451 < Pose_Y <= 790:
                                region = 'bottom_right'
                            elif 320 < Pose_X <= 970 and 250 < Pose_Y <= 550:
                                region = 'central_area'
                            else:
                                if generate_OTHERS:
                                    region = 'OTHERS'
                                else:
                                    continue

                        if '/left/' in picture_name:
                            hand = 'left'
                        elif '/right/' in picture_name:
                            hand = 'right'
                        else:
                            hand = 'NO_HANDS'

                        if not pd.isnull(selected_df.loc[i, 'GT_Z ']):
                            z = selected_df['GT_Z '][i]
                            z = 'Z' + str(z)
                        else:
                            z = 'NO_Zinfo'

                        y1 = int(min(672, max(0, Pose_Y-64)))
                        y2 = y1+128
                        x1 = int(min(1152, max(0, Pose_X-64)))
                        x2 = x1+128
                        image_C1 = cv2.imread(os.path.join(original_imgs[list_index], picture_name))
                        cropped_C1 = image_C1[y1:y2, x1:x2]
                        if sequence_Data:
                            save_path = os.path.join(Curr_save, os.path.split(picture_name)[1])    # for Sequence Data
                        else:
                            save_path = os.path.join(Curr_save, person, Detection_Type, region, hand, z, os.path.split(picture_name)[1])
                        save_dir = os.path.dirname(save_path)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        # For multiple detections in one image. I've seen some ones have four detections and I know the
                        # code sucks, you can upgrade it.
                        # if os.path.isfile(save_path):
                        #     save_path = save_path.replace('_c1', '-02_c1')
                        #     if os.path.isfile(save_path):
                        #         save_path = save_path.replace('-02_c1', '-03_c1')
                        #         if os.path.isfile(save_path):
                        #             save_path = save_path.replace('-03_c1', '-04_c1')
                        #             if os.path.isfile(save_path):
                        #                 save_path = save_path.replace('-04_c1', '-05_c1')

                        counter = 1
                        check_path = save_path.replace('_c1_', '_{}_c1_')
                        while os.path.exists(save_path):
                            save_path = check_path.format(counter)
                            counter += 1

                        cv2.imwrite(save_path, cropped_C1)

                        image_C2 = cv2.imread(os.path.join(original_imgs[list_index], picture_name.replace('c1', 'c2')))
                        cropped_C2 = image_C2[y1:y2, x1:x2]
                        save_path = save_path.replace('c1', 'c2')
                        cv2.imwrite(save_path, cropped_C2)
                        print(save_path)

                        # You can comment codes below, it may slow the code down
                        # cv2.imshow("cropped_C1", cropped_C1)
                        # cv2.imshow("cropped_C2", cropped_C2)
                        # cv2.waitKey(10)


if __name__ == '__main__':
    main()
