"""Local football image analysis. Launch with scripts/setup.py app."""
from collections import Counter
import hashlib
import logging
from pathlib import Path
import sys

import streamlit as st

# Streamlit executes this file as a script, so expose the repository package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports_analyzer.image_io import decode_image, encode_png
from sports_analyzer.inference import Detector
from sports_analyzer.visualization import draw_detections

LOGGER = logging.getLogger(__name__)
MODEL_PATH = ROOT / 'models/yolo26n.pt'


def main():
    st.set_page_config(page_title='Sports Analyzer', page_icon='⚽', layout='wide')
    st.title('Sports Analyzer')
    st.write('Upload a football image and explore the detected objects.')

    with st.sidebar:
        st.header('Analysis settings')
        st.selectbox('Sport', ['Football'], disabled=True)
        imgsz = st.selectbox('Model input size', [640, 1280])
        confidence = st.slider('Minimum confidence', 0.05, 0.95, 0.25, 0.05)
        st.caption('Model: YOLO26n · pretrained · CPU')
        st.caption('Detects people and sports balls. Player roles, teams and '
                   'possession are not yet identified.')

    uploaded = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'],
                                max_upload_size=10)
    st.caption('JPEG or PNG · up to 10 MiB and 20 megapixels')
    if uploaded is None:
        for key in ('upload_id', 'image', 'result'):
            st.session_state.pop(key, None)
        st.info('Choose an image to start the analysis.')
        return

    data = uploaded.getvalue()
    upload_id = hashlib.sha256(data).hexdigest()
    if st.session_state.get('upload_id') != upload_id:
        st.session_state.pop('result', None)
        st.session_state.pop('image', None)
        st.session_state.pop('upload_id', None)
        try:
            st.session_state.image = decode_image(data)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state.upload_id = upload_id

    image = st.session_state.image
    request_key = (upload_id, imgsz, confidence)
    previous = st.session_state.get('result')
    if previous is not None and previous['key'] != request_key:
        st.session_state.pop('result', None)

    original_column, result_column = st.columns(2)
    with original_column:
        st.subheader('Original')
        st.image(image, width='stretch')
        st.caption(f'{image.width} × {image.height} pixels')

    if not MODEL_PATH.is_file():
        st.error('Model weights are missing. Run '
                 '`py -3.12 scripts/setup.py` from the project root, then reload this page.')
        return

    if st.button('Analyse image', type='primary'):
        st.session_state.pop('result', None)
        try:
            with st.spinner('Detecting objects …'):
                # One model per browser session; no shared mutable predictor.
                if 'detector' not in st.session_state:
                    st.session_state.detector = Detector(MODEL_PATH, device='cpu')
                boxes, timings = st.session_state.detector.predict_image(
                    image, imgsz=imgsz, confidence=confidence
                )
                annotated = draw_detections(image, boxes)
                st.session_state.result = {
                    'key': request_key, 'png': encode_png(annotated),
                    'boxes': boxes, 'timings': timings,
                }
        except Exception:
            LOGGER.exception('Image analysis failed')
            st.session_state.pop('detector', None)
            st.error('Analysis failed. See the terminal for details. '
                     'Check the model and environment, then try again.')
            return

    result = st.session_state.get('result')
    with result_column:
        st.subheader('Result')
        if result is None:
            st.info('Click “Analyse image” to see the result.')
        else:
            st.image(result['png'], width='stretch')
            st.download_button('Download annotated image', data=result['png'],
                               file_name='football-annotated.png', mime='image/png',
                               on_click='ignore')

    if result is not None:
        boxes = result['boxes']
        count_column, time_column = st.columns(2)
        count_column.metric('Detected objects', len(boxes))
        time_column.metric('Prediction time', f"{result['timings']['predict_call_ms']:.0f} ms")
        st.caption('Prediction time includes preprocessing and postprocessing; excludes model loading, '
                   'drawing and download. Single measurement, not a benchmark.')
        if not boxes:
            st.info('No objects detected above the selected confidence threshold.')
        else:
            counts = Counter(box.label for box in boxes)
            st.write(' · '.join(f'{label}: {count}' for label, count in sorted(counts.items())))
            with st.expander('Detection details'):
                st.dataframe([
                    {'Class': box.label, 'Score': round(box.score, 3),
                     'x_min': round(box.xyxy[0], 1), 'y_min': round(box.xyxy[1], 1),
                     'x_max': round(box.xyxy[2], 1), 'y_max': round(box.xyxy[3], 1)}
                    for box in boxes
                ], hide_index=True)


if __name__ == '__main__':
    main()
