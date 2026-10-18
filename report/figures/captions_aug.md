# Augmentation figure caption (draft)

## aug_examples

**Caption.** Training-time augmentation for the pretrained models (default setting). Column 1: the stored image (shorter side 288 px); column 2: the deterministic validation/test input (resize + 224 × 224 centre crop); columns 3-7: independent random draws of the training transform - random resized crop (50-100 % of the area, aspect 3/4-4/3) to 224 × 224, horizontal flip (p = 0.5), rotation (±15°), colour jitter (brightness/contrast/saturation 0.2, hue 0.02) and random erasing (p = 0.25; the erased box takes the ImageNet mean colour); rotation leaves black corners. The scratch CNN uses the same transform at 128 × 128 with crops of 60-100 % of the area.

**Alt text.** Two rows (a cat and a dog) of seven images: the original photo, its centre crop and five randomly cropped, flipped, rotated and colour-shifted versions, some with a grey erased rectangle.
