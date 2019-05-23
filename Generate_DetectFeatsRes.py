import cv2, os, fnmatch, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model
import keras.backend as K


'''
Script: Visualize activation maps for a specific image input
Output: One image for each convolutional layer.
'''


def main():
    # ==========================================Please Modify Information Below=========================================
    MODEL_FILE = r'Models\detection_D1.5_L\weights_epoch1500_valLoss161.hdf5'
    srcDir = r'VisExps\ROI_imgs\20190311_134926\TF_classification-25-2.558'
    resultPath = r'VisExps\DetectFeatsRes\detection_D1.5_L'

    # for D1.5L and D1.7L
    layer_names = ['conv_11', 'conv_12', 'conv_13', 'conv_21', 'conv_22', 'conv_23', 'conv_31', 'conv_32', 'conv_33',
                   'conv_41', 'conv_42', 'conv_43', 'conv_51', 'conv_52', 'conv_53']
    # for D2.4S
    # layer_names = ['conv_11', 'conv_12', 'conv_21', 'conv_22', 'conv_31', 'conv_32', 'conv_41', 'conv_42', 'conv_51']

    normalize_methods = ['mean_individual', 'max-min_individual', 'mean_layer', 'max-min_layer']
    normalize_method = normalize_methods[3]
    save_colorBar = False
    get_csv = True

    # ============================================ Start from Loading Model=============================================
    model = load_model(MODEL_FILE, compile=False)
    plot_model(model, to_file=os.path.join(os.path.dirname(MODEL_FILE), 'detection_model.png'))
    model.summary()

    time_start = time.time()
    first_img = True
    if get_csv:
        csv_path = os.path.join(srcDir, 'img_info.csv')
        if not os.path.exists(csv_path):
            col_val = ['img_path', 'imgResDir']
            df = pd.DataFrame(columns=col_val)
            df = get_img_csv(srcDir, resultPath, df)
            df.to_csv(csv_path, index=False)
        else:
            df = pd.read_csv(csv_path)

        # if you have the index information, you can resume processing by setting the start index
        for index in range(len(df.index)):
            img = cv2.imread(df.loc[index, 'img_path'], 0)              # input img
            imgResDir = df.loc[index, 'imgResDir']
            print('Processed: {}/{}'.format(index, len(df.index)))
            process_img(img, imgResDir, first_img, layer_names, model, normalize_method, save_colorBar)
            first_img = False
    else:
        for root, dirnames, filenames in os.walk(srcDir):
            for filename in sorted(filenames):
                if not fnmatch.fnmatch(filename, '*.png'):
                    continue
                # Read input image and create a folder to save its results
                img = cv2.imread(os.path.join(root, filename), 0)
                mid_path = root.replace(srcDir + '\\', '')
                imgResDir = os.path.join(resultPath, mid_path, filename[:-4])
                cv2.imwrite(os.path.join(imgResDir, filename), img)
                print(imgResDir)

                # Call process_img function
                process_img(img, imgResDir, first_img, layer_names, model, normalize_method, save_colorBar)
                first_img = False
    time_total = time.time() - time_start
    print('Totaltime: '.format(time_total))


def process_img(img, imgResDir, first_img, layer_names, model, normalize_method, save_colorBar):
    if not os.path.exists(imgResDir):
        os.makedirs(imgResDir)
    if 'float' in str(img.dtype):
        img = (255 * img).astype('uint8')
    img = np.expand_dims(np.expand_dims(img, axis=0), axis=3)

    # ==================================Get Activation maps for each channel of each layer==============================
    activation_maps = get_activations(model, img, layer_name=layer_names)   # For Given Layer_names list
    # activation_maps = get_activations(model, img)                         # All of the layers in the model

    if first_img:
        for i in range(len(activation_maps)):
            print(activation_maps[i].shape)

    # =========================================== Combine the channel_images from same layer together===================
    images_per_row = activation_maps[0].shape[-1]
    for layer_name, layer_activation in zip(layer_names, activation_maps):
        n_features = layer_activation.shape[-1]
        img_size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((img_size * n_cols, images_per_row * img_size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                # Post-process the feature
                if normalize_method == 'mean_individual':
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                elif normalize_method == 'max-min_individual':
                    if display_grid.max() != 0:
                        channel_image = (channel_image - channel_image.min()) / (
                                channel_image.max() - channel_image.min())
                    else:
                        raise Exception('display_grid.man() returned 0, the img is in: {}'.format(imgResDir))
                    channel_image *= 255
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                display_grid[col * img_size: (col + 1) * img_size, row * img_size: (row + 1) * img_size] = channel_image
                plt.imshow(display_grid, cmap='viridis')

        if normalize_method == 'mean_layer':
            display_grid -= display_grid.mean()
            display_grid /= display_grid.std()
            display_grid *= 64
            display_grid += 128
            display_grid = np.clip(display_grid, 0, 255).astype('uint8')
        elif normalize_method == 'max-min_layer':
            if display_grid.max() != 0:
                display_grid = (display_grid - display_grid.min()) / (display_grid.max() - display_grid.min())
            else:
                raise Exception('display_grid.man() returned 0, the img is in: {}'.format(imgResDir))
            display_grid *= 255
            display_grid = np.clip(display_grid, 0, 255).astype('uint8')

        scale_height = int(n_features / images_per_row)
        scale_width = int(n_features * img_size) // display_grid.shape[1]
        resized = cv2.resize(display_grid, None, fx=scale_width, fy=scale_height, interpolation=cv2.INTER_AREA)

        # =================== Save Image
        if not save_colorBar:
            plt.imsave(os.path.join(imgResDir, normalize_method + '_' + layer_name + '.png'), resized, cmap='viridis')
        else:
            fig, ax = plt.subplots()
            im = ax.imshow(resized, cmap='viridis')
            fig.colorbar(im, orientation='horizontal', pad=0.05)
            plt.imshow(resized, cmap='viridis')
            plt.savefig(os.path.join(imgResDir, 'Bar' + normalize_method + '_' + layer_name + '.png'))
        plt.close()


def get_activations(model, model_inputs, layer_name=None):
    activations = []
    inp = model.input

    model_multi_inputs = True
    if not isinstance(inp, list):
        inp = [inp]
        model_multi_inputs = False

    if model_multi_inputs:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    if layer_name is None:
        outputs = [layer.output for layer in model.layers if layer.name != 'model_input']
    else:
        outputs = [layer.output for layer in model.layers if layer.name in layer_name]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)

    return activations


def get_img_csv(folder, resultPath, df):
    """
    Construct a .csv file containing the imp_path and result_path
    :param folder: the path to the folder where you want to start exploration
    :param resultPath: the result big folder
    :return: df: a pandas.DataFrame
    """
    print('Generating image look up table.')
    idx = 0
    for root, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if not fnmatch.fnmatch(filename, '*.png'):
                continue
            df.loc[idx, 'img_path'] = os.path.join(root, filename)
            mid_path = root.replace(folder + '\\', '')
            mid_path = mid_path.replace(folder + '/', '')     # added for linux OS
            df.loc[idx, 'imgResDir'] = os.path.join(resultPath, mid_path, filename[:-4])
            idx += 1
    print('Successfully generated image LUT.')
    return df


if __name__ == '__main__':
    main()