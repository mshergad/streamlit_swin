# import use case specific packages
# ideally you want all methods to be in a separate helper file
import streamlit as st
import helper as hlp

# add fiddler cluster details on which your CV model is uploaded
URL = 'https://demo4amfam.dev.fiddler.ai'
API_KEY = 'l4V1Rxve9sb7Iugacc72xWsPSr1uqw3uxK5YDok0sbw'
ORG_ID = 'demo4amfam'

# add project, model, and dataset names used on fiddler platform
PROJECT_ID = 'swin_magpie_32'
DATASET_ID = 'magpie_data'
MODEL_ID = 'classification_magpie'

# add column name of the dataframe where image is stored as string, output col name and labels
DF_IMAGE_COL_NAME = 'image'
OUTPUT_COLS = ['predicted_magpie']
OUTPUT_LABELS = ['magpie', 'not magpie']

# location of pandas dataframe
DATA_FILE = 'magpie_data_real.pkl'

# load the dataframe
df = hlp.load_data(DATA_FILE, preprocess=True)


#streamlit specific variables do not edit
FILE_ID = 'file_id'
IMAGE_ID = 'image_id'
if IMAGE_ID not in st.session_state:
    st.session_state[IMAGE_ID] = 2

STYLE_POOLING = 'style_pooling'
if STYLE_POOLING not in st.session_state:
    st.session_state[STYLE_POOLING] = 'Max'

STYLE_TRANS = 'style_trans'
if STYLE_TRANS not in st.session_state:
    st.session_state[STYLE_TRANS] = 'Transparency'

CONTRAST = 'contrast'
if CONTRAST not in st.session_state:
    st.session_state[CONTRAST] = 0.3

SEARCH_TYPE = 'search_type'
if SEARCH_TYPE not in st.session_state:
    st.session_state[SEARCH_TYPE] = 'All Images'

print(st.session_state)


style_pooling = hlp.POOL_MAX if st.session_state[STYLE_POOLING] == 'Max' else hlp.POOL_SUM
style_trans = True if st.session_state[STYLE_TRANS] == 'Transparency' else False

exp = hlp.Exp()

i = int(st.session_state[IMAGE_ID])
df_explain = df[i:i+1]


def search_images(current_index):
    # was + or - clicked?
    current_index = int(current_index)
    direction = st.session_state[IMAGE_ID] - current_index

    if abs(direction) != 1 or st.session_state[SEARCH_TYPE] == 'All Images':
        return

    if direction == 1:
        search_df = df.iloc[current_index+1:]

        for j, x in search_df.iterrows():
            if x['predicted'] == OUTPUT_LABELS[0] and st.session_state[SEARCH_TYPE] == 'Magpie':
                st.session_state[IMAGE_ID] = j
                return

            if x['predicted'] == OUTPUT_LABELS[1] and st.session_state[SEARCH_TYPE] == 'Not Magpie':
                st.session_state[IMAGE_ID] = j
                return
    else:
        search_df = df.iloc[:current_index]
        for j, x in search_df.iloc[::-1].iterrows():
            if x['predicted'] == OUTPUT_LABELS[0] and st.session_state[SEARCH_TYPE] == 'Magpie':
                st.session_state[IMAGE_ID] = j
                return

            if x['predicted'] == OUTPUT_LABELS[1] and st.session_state[SEARCH_TYPE] == 'Not Magpie':
                st.session_state[IMAGE_ID] = j
                return

    # if couldn't find one, leave index unchanged
    st.session_state[IMAGE_ID] = current_index


exp.image_array = hlp.unpack_images(df_explain, DF_IMAGE_COL_NAME)

if FILE_ID in st.session_state and st.session_state[FILE_ID] is not None:
    df_explain = hlp.prepare_df(st.session_state[FILE_ID], DF_IMAGE_COL_NAME)
    exp.attributions, exp.predictions, exp.inferred_class_index, exp.path_preds = hlp.explain(df_explain, URL, ORG_ID,
                                                                                              API_KEY, PROJECT_ID,
                                                                                              MODEL_ID)
    exp.file_array = hlp.unpack_file(st.session_state[FILE_ID])
else:
    exp.attributions, exp.predictions, exp.inferred_class_index, exp.path_preds = hlp.explain(df_explain, URL, ORG_ID,
                                                                                              API_KEY, PROJECT_ID,
                                                                                              MODEL_ID)

st.image('poweredby.png', width=250)

st.title('Image Explainability with Fiddler AI')
with st.expander("Introduction"):
    st.write('This web app demonstrates a Fiddler API integration and uses the Integrated Gradients [IG] attribution method to produce helpful explanations of deep learning models with image inputs. The IG calculation is performed on a Fiddler cluster and is requested via API. The web app sends an image from a local collection and Fiddler returns an attribution matrix and other related details. In order to produce this functionality, the user\'s model form factor is specified, and the artifact is uploaded via APIs in the Fiddler client library.')


col1, mid, col2 = st.columns([10, 5, 10])


with col1:
    st.subheader('Dataframe row selector')
    st.number_input(f'Select an image number (0-{len(df) - 1})',
                    key=IMAGE_ID, min_value=0, max_value=len(df),
                    on_change=search_images, args=[st.session_state[IMAGE_ID]])
    st.radio('Image Selector Finds:', ['All Images', 'Magpie', 'Not Magpie'], key=SEARCH_TYPE)

with mid:
    st.markdown("<h1 style='text-align: center; color: red;'>OR</h1>", unsafe_allow_html=True)
with col2:
    st.subheader('Image file uploader')
    st.file_uploader('Image Upload', type=['png', 'jpg', 'jpeg'], key=FILE_ID)

col1, _, col2 = st.columns([10, 1, 10])

with col1:

    if FILE_ID in st.session_state and st.session_state[FILE_ID] is not None:
        st.subheader(f'Input File')
        st.caption('Uploaded image data provided to the model resulting in the predictions to the right.')
    else:
        st.subheader(f'Input Image (Example {i})')
        st.caption('Source "test" data provided to the model resulting in the predictions to the right.')

    hlp.plot_input_image(exp)


output_labels = OUTPUT_LABELS
with col2:
    st.subheader('Predictions')
    st.caption('Predictions of the model; "actual" indicates the provided ground-truth label.')
    for i, col in enumerate(output_labels):
        p = exp.predictions if i == 0 else 1 - exp.predictions

        predicted_color = 'blue' if output_labels[exp.inferred_class_index] == df_explain.iloc[0]['predicted'] else 'red'
        if i == exp.inferred_class_index:
            color = predicted_color
            weight = 'font-weight:bold;'
        else:
            color = 'black'
            weight = ''

        spacing = 15 if i == 0 else 15

        actual = '  (actual)' if df_explain.iloc[0]['predicted'] == output_labels[i] else ''

        st.markdown(f'<div style="padding-top:{spacing}px"/>{col}<BR/><span style="font-size:22pt; {weight}color:{color}">{p:0.2f}</span><span style="font-size:22pt"><span>{actual}</span>', unsafe_allow_html=True)

st.markdown('---')

col1, _, col2 = st.columns([10, 1, 10])

with col1:
    st.subheader('Explanation for magpie hypothesis')
    st.caption("Colored overlay indicates the regions that influence the model's prediction. (blue/red:positive/negative)")
    hlp.plot_processed_attributions(exp, transf=style_trans, pool=style_pooling,
                                threshold=st.session_state[CONTRAST],
                                transparency_contrast=st.session_state[CONTRAST])

with col2:
    st.radio(f'Display Style – Attribution Pooling', ['Max', 'Sum'], key=STYLE_POOLING, )
    st.radio(f'Display Style – Overlay', ['Transparency', 'Threshold'], key=STYLE_TRANS)
    st.slider(f'Contrast/Threshold (Default=0.3)', min_value=0., max_value=1., key=CONTRAST)

with st.expander("Expert Details"):

    col1, _, col2 = st.columns([4, 1, 4])

    with col1:
        st.subheader('Raw Attributions')
        hlp.plot_raw_attributions(exp)

    with col2:
        st.subheader('Predictions Along IG Path')
        hlp.plot_predictions(exp, OUTPUT_COLS)