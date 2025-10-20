# FoodSynthetics: A Dataset for AI-Generated Food Image Classification
This is the repository for the FoodSynthetics dataset
<img width="7500" height="2250" alt="ai_samples_grid_1" src="https://github.com/user-attachments/assets/c8f10503-abd0-41ba-8908-88957df49180" />
<img width="7500" height="2250" alt="real_samples_grid" src="https://github.com/user-attachments/assets/1156e0f6-9205-4b77-8f91-d0817a608776" />


## Dataset collection
The real class data was sourced from the [Food101 dataset](https://www.kaggle.com/datasets/dansbecker/food-101), which contains photographs across 101 distinct categories.

Synthetic images were generated using Flux.1 and Midjourney V7, both diffusion-based models capable of producing highly realistic visuals. Because these models rely on text-to-image prompting, careful prompt engineering was essential to maximize realism. Prompts were derived from each Food101 category, with multiple variations per class to enhance visual diversity and improve classifier generalization.

The final dataset comprises approximately 32,000 images, evenly balanced between real and synthetic classes.

### Dataset Access
The Dataset is available for download via [Google Drive](https://drive.google.com/file/d/17IrR5QdqbUaS63hjsSdpzJyn-F8RUpjY/view?usp=sharing). The dataset consists of 32,830 balanced samples of real and synthetic food images. The link also provide a split that was used to train various models on the paper.

## Classification and CAM
Several convolutional neural network (CNN) architectures were trained to distinguish between real and synthetic food images. The models evaluated include ResNet (18 and 50), EfficientNet (B2, B3), and ConvNeXt-Tiny, representing a range of modern CNN backbones with varying depth and computational complexity.

Training was conducted using the balanced dataset, split into training, validation, and test sets. Standard data augmentation techniques were applied to improve model robustness, while additional transformations such as JPEG compression were selectively applied to synthetic images to increase detection difficulty and prevent overfitting to generation artifacts.
```
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 2.0), p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.ImageCompression(quality_lower=75, quality_upper=90, p=0.4),  # JPEG compression
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
val_test_transform = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
```

Among the tested models, ConvNeXt-Tiny achieved the highest classification accuracy of 96.79%, demonstrating strong generalization and discriminative ability between real and AI-generated content.

Grad-CAM visualizations were used to interpret model predictions, revealing which image regions contributed most to the classification decision and providing insights into the learned visual cues distinguishing real from synthetic food imagery.
<img width="3600" height="600" alt="AI_0b75c6aab5ad_TN_TN_TN_TN_TN" src="https://github.com/user-attachments/assets/cc692f10-342c-4155-b554-dba47e3e0e6b" />
<img width="3600" height="600" alt="Real_02cdc409a1f9_TP_TP_TP_TP_TP" src="https://github.com/user-attachments/assets/5ca3dd10-489e-481b-92af-6eb103850388" />

