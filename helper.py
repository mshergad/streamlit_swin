import streamlit as st
import pickle as pk
import base64
import io
import numpy as np
from PIL import Image
import fiddler as fdl
import matplotlib.pyplot as plt
import torch
import pandas as pd

TRANSFORM_FILE = 'orig_trans.pkl'


POOL_MAX = 'pool_max'
POOL_SUM = 'pool_sum'


@st.cache
def load_data(data_file, preprocess=False):
    with open(data_file, 'rb') as f:
        dfl = pk.load(f)
        if preprocess:
            # optional preprocessing steps for your own df
            dfl.loc[dfl['predicted'] != 'magpie', 'predicted'] = 'not magpie'
        return dfl

@st.cache
def trans():
    with open(str(TRANSFORM_FILE), 'rb') as f:
        return pk.load(f)


tr = trans()

def downsample_sum_pool(image, bin_size):
    input_size = image.shape[0]
    output_size = input_size // bin_size
    return image.reshape(output_size, bin_size, output_size, bin_size).sum(3).sum(1)


def sum_pool(image, bin_size):
    return upsample(downsample_sum_pool(image, bin_size), bin_size)


def downsample_max_pool(image, bin_size):
    input_size = image.shape[0]
    output_size = input_size // bin_size
    return image.reshape(output_size, bin_size, output_size, bin_size).max(3).max(1)


def upsample(image, bin_size):
    return np.repeat(image, bin_size, axis=1).repeat(bin_size, axis=0)


def max_pool(image, bin_size):
    return upsample(downsample_max_pool(image, bin_size), bin_size)


def map_colors(x):
    return plt.cm.bwr_r(x)


def add_trans(image, contrast=1.):
    mn = image.min()
    mx = image.max()

    max_scale = max(np.abs([mn, mx]))
    g2 = 0.5 + image / (2 * max_scale)
    g_abs = np.abs((image / max_scale))
    g3 = map_colors(g2)
    g3[:, :, 3] = g_abs ** contrast
    return g3


def add_threshold_trans(img, threshold=0.3):
    # takes a one-channel image and converts it to RGB+A where below threshold pixels are transparent, above are opaque.
    thresh_image = add_trans(img)
    thresh_image[:, :, 3] = np.ones(thresh_image[:, :, 3].shape) * (thresh_image[:, :, 3] > threshold)
    return thresh_image


def convert_image(image):
    image = tr(image)
    image.requires_grad = True
    return image

# to retrieve original image before preprocessing transformations are applied
def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor

def swap_axis(image):
    temp = image.swapaxes(0, 2)
    final = temp.swapaxes(0, 1)
    return final


@st.cache
def prepare_df(uploaded_file, column_name):
    data_encoded = base64.b64encode(uploaded_file.getvalue()).decode('ASCII')
    return pd.DataFrame({column_name: [data_encoded], 'predicted': 'Not Sure'})


@st.cache
def unpack_images(input_df, column_name):
    files = [base64.b64decode(x) for x in input_df[column_name].values]
    orig_images = [Image.open(io.BytesIO(file)).convert('RGB') for file in files]
    conv = convert_image(orig_images[0])
    inv_swapped2 = inverse_normalize(conv.detach())
    final = swap_axis(inv_swapped2)
    return final


@st.cache
def unpack_file(uploaded_file):
    files = [uploaded_file.getvalue()]
    orig_images = [Image.open(io.BytesIO(file)).convert('RGB') for file in files]
    conv = convert_image(orig_images[0])
    inv_swapped2 = inverse_normalize(conv.detach())
    final = swap_axis(inv_swapped2)
    return final


def plot_predictions(exp, output_cols):

    plt.figure(figsize=[6, 4])
    fig, ax = plt.subplots()

    for j, output_name in enumerate(output_cols):
        ax.plot(np.linspace(0, 1, len(exp.path_preds)), exp.path_preds, label=output_name)

    ax.set_title('Fractional Distance from Baseline to Explain Point', fontsize=16)
    ax.set_xlabel(rf'$\alpha$ ({len(exp.path_preds)} steps)', fontsize=14)
    ax.set_ylabel('Model Output', fontsize=14)
    ax.legend(fontsize=12, loc='lower center')

    st.pyplot(fig)


def plot_input_image(ex):
    fig, ax = plt.subplots()
    if ex.file_array is None:
        ax.imshow(ex.image_array, aspect=ex.aspect)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    else:
        print("file print")
        ax.imshow(ex.file_array, aspect=ex.aspect)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    st.pyplot(fig)


def plot_raw_attributions(ex):

    mn = ex.attributions.min()
    mx = ex.attributions.max()
    max_scale = max(np.abs([mn, mx]))
    fig, ax = plt.subplots()
    ax.imshow(ex.attributions, aspect=ex.aspect, cmap='bwr_r', vmin=-max_scale, vmax=max_scale)
    st.pyplot(fig)


def plot_processed_attributions(ex, overlay=True, pool=POOL_MAX, transf=False,
                                n_pool=4, transparency_contrast=0.3, threshold=0.3,):

    aspect = ex.aspect
    attributions = ex.attributions
    image_array = ex.image_array
    file_array = ex.file_array

    fig, ax = plt.subplots()

    if overlay and file_array is None:
        ax.imshow(image_array, aspect=aspect)
    elif overlay and file_array is not None:
        ax.imshow(file_array, aspect=aspect)

    if pool is POOL_SUM:
        if not overlay:
            img = sum_pool(attributions, n_pool)
            mn = img.min()
            mx = img.max()
            max_scale = max(np.abs([mn, mx]))
            ax.imshow(img, aspect=aspect, cmap='bwr_r', vmin=-max_scale, vmax=max_scale)

        elif transf:
            sum_pooled_w_transparency = add_trans(sum_pool(attributions, n_pool), contrast=transparency_contrast)

            ax.imshow(sum_pooled_w_transparency, aspect=aspect)

        else:
            ax.imshow(add_threshold_trans(sum_pool(attributions, n_pool), threshold=threshold),
                      aspect=aspect, vmin=0, vmax=1)

    elif pool is POOL_MAX:
        if not overlay:
            mn = attributions.min()
            mx = attributions.max()
            max_scale = max(np.abs([mn, mx]))
            ax.imshow(max_pool(attributions, n_pool), aspect=aspect, cmap='bwr_r', vmin=-max_scale, vmax=max_scale)

        elif transf:
            # ax.imshow(add_trans(attributions, 0.1))
            ax.imshow(add_trans(max_pool(attributions, n_pool), contrast=transparency_contrast), aspect=aspect)

        else:
            ax.imshow(add_threshold_trans(max_pool(attributions, n_pool), threshold=threshold), aspect=aspect)

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    st.pyplot(fig)


@st.cache
def explain(input_df, URL, ORG_ID, API_KEY, PROJECT_ID, MODEL_ID):
    api = fdl.FiddlerApi(url=URL, org_id=ORG_ID, auth_token=API_KEY)
    full_response = api.run_explanation(PROJECT_ID, MODEL_ID, input_df, explanations='ig_flex')

    atts = np.asarray(full_response.explanations['predicted_magpie'].attributions['contents'][0]['image-attributions'])
    paths_preds = full_response.explanations['predicted_magpie'].attributions['contents'][0]['path-preds']
    preds = np.asarray(full_response.explanations['predicted_magpie'].misc['model_prediction'])
    index = 0 if preds > 0.5 else 1
    #
    # atts = full_response.explanations['predicted_cola_demand'].attributions['ig_explanation']
    # atts = np.asarray(atts[index]['input_image'])
    # atts = atts.transpose([1, 2, 0]).sum(2)
    #
    # device = np.asarray(full_response.explanations['predicted_cola_demand'].attributions['device_type'])
    # time = np.asarray(full_response.explanations['predicted_cola_demand'].attributions['time'])

    return atts, preds, index, paths_preds  #, preds, index, device, time

class Exp:
    def __init__(self):
        self.aspect = None
        self.image_array = None
        self.file_array = None
        self.attributions = None
        self.predictions = None
        self.device = None
        self.time = None
        self.inferred_class_index = 0
        self.path_preds = None



