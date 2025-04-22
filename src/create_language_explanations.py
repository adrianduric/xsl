#Imported modules
from config import params
from data_mgmt_hyperkvasir_multimodal import prepare_data_hyperkvasir_multimodal

import os
from tqdm import tqdm
import torch
from transformers import pipeline


def create_language_explanations():

    # Initializing complete dataset
    complete_dataset, complete_dataloader = prepare_data_hyperkvasir_multimodal(seed=None, augmented_data=False, split=False, batch_size=1)

    # Initializing model to create explanations with
    pipe = pipeline(
        "image-text-to-text",
        model="google/gemma-3-4b-pt",
        device=params["device"],
        torch_dtype=torch.bfloat16
    )

    # Prompt used to generate explanations
    prompt="<start_of_image> This image is from a colonoscopy examination in the gastrointestinal tract, and shows:"

    print("Creating language explanations for all images...")

    for image, text, label, img_path, txt_path in tqdm(complete_dataloader):

        # Generate explanation
        output = pipe(
            img_path,
            text=prompt,
            max_new_tokens=50
        )
        explanation = output[0]["generated_text"]

        # Save explanation in .txt file
        explanation_filename = os.path.join(os.getcwd(), os.path.relpath(txt_path[0]))
        os.makedirs(os.path.dirname(explanation_filename), exist_ok=True)
        
        with open(explanation_filename, 'w') as file:
            file.write(explanation)


if __name__ == "__main__":
    create_language_explanations()
