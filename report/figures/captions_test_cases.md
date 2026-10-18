# Part (e) test-set figure captions (draft)

Add what each hard image shows (see CONTEXT.md §4e).

## test_cases

**Caption.** Test-set examples of the final model. (a) The 6 test images with the least confident predictions; a visual check of all 500 test images found every prediction correct, including these. (b) The most confident cats and dogs. p(dog) is the final model's probability (3-seed average with flip TTA); because training used label smoothing 0.1, confident predictions sit near 0.05 (cat) and 0.95 (dog) rather than 0 and 1.

## test_confidence

**Caption.** Distribution of the final model's p(dog) over the 500 test images (log-scale counts). Only 3 images fall between 0.2 and 0.8; the rest are confident. Label smoothing (0.1) keeps confident predictions near 0.05 and 0.95.
