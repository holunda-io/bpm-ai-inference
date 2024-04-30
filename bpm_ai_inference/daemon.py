import logging
import os

from bpm_ai_core.util.rpc import create_remote_object_daemon

from bpm_ai_inference.classification.transformers_text_classifier import TransformersClassifier
from bpm_ai_inference.image_classification.transformers_image_classifier import TransformersImageClassifier
from bpm_ai_inference.llm.llama_cpp.llama_chat import ChatLlamaCpp
from bpm_ai_inference.ocr.tesseract import TesseractOCR
from bpm_ai_inference.pos.spacy_pos_tagger import SpacyPOSTagger
from bpm_ai_inference.question_answering.pix2struct_vqa import Pix2StructVQA
from bpm_ai_inference.question_answering.transformers_docvqa import TransformersDocVQA
from bpm_ai_inference.question_answering.transformers_qa import TransformersExtractiveQA
from bpm_ai_inference.speech_recognition.faster_whisper import FasterWhisperASR
from bpm_ai_inference.token_classification.gliner_token_classifier import GlinerTokenClassifier
from bpm_ai_inference.token_classification.transformers_token_classifier import TransformersTokenClassifier
from bpm_ai_inference.translation.easy_nmt.easy_nmt import EasyNMT

remote_classes = [
    TransformersClassifier,
    TransformersImageClassifier,
    TransformersTokenClassifier,
    GlinerTokenClassifier,
    TesseractOCR,
    SpacyPOSTagger,
    Pix2StructVQA,
    TransformersDocVQA,
    TransformersExtractiveQA,
    FasterWhisperASR,
    EasyNMT,
    ChatLlamaCpp
]

if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s: %(message)s', level=int(os.getenv('LOG_LEVEL', str(logging.INFO))))

    daemon = create_remote_object_daemon(
        host=os.getenv('DAEMON_HOST', '0.0.0.0'),
        port=int(os.getenv('DAEMON_PORT', 6666)),
        instance_strategy=os.getenv('INSTANCE_STRATEGY', 'memory_limit'),
        max_memory=int(os.getenv('SOFT_MEMORY_LIMIT', 8_589_934_592))
    )

    for c in remote_classes:
        daemon.register_class(c)

    daemon.serve()
