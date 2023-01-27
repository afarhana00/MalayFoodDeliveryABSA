from pyabsa.core.atepc.prediction.aspect_extractor import AspectExtractor
from pyabsa import ATEPCCheckpointManager


def prediction(text):
    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint="checkpoints\lcf_atepc_custom_dataset_cdw_apcacc_64.74_apcf1_64.85_atef1_26.79",auto_device=False)

    predicted_text = aspect_extractor.extract_aspect(inference_source=[text],
                                            save_result=True,
                                            print_result=True,  # print the result
                                            pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                            )

    return predicted_text[0]['aspect'], predicted_text[0]['sentiment'], predicted_text[0]['confidence']