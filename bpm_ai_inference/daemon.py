import logging
import os

from bpm_ai_core.util.remote_object import create_remote_object_daemon

from bpm_ai_inference.classification.transformers_classifier import TransformersClassifier
from bpm_ai_inference.ocr.tesseract import TesseractOCR
from bpm_ai_inference.pos.spacy_pos_tagger import SpacyPOSTagger
from bpm_ai_inference.question_answering.pix2struct_vqa import Pix2StructVQA
from bpm_ai_inference.question_answering.transformers_docvqa import TransformersDocVQA
from bpm_ai_inference.question_answering.transformers_qa import TransformersExtractiveQA
from bpm_ai_inference.speech_recognition.faster_whisper import FasterWhisperASR
from bpm_ai_inference.translation.easy_nmt.easy_nmt import EasyNMT

remote_classes = [
    TransformersClassifier,
    TesseractOCR,
    SpacyPOSTagger,
    Pix2StructVQA,
    TransformersDocVQA,
    TransformersExtractiveQA,
    FasterWhisperASR,
    EasyNMT
]

if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    daemon = create_remote_object_daemon(
        host=os.environ.get('DAEMON_HOST', '0.0.0.0'),
        port=int(os.environ.get('DAEMON_PORT', 6666))
    )

    for c in remote_classes:
        daemon.register_class(c)

    daemon.serve()
