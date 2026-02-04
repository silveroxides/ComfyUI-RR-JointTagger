# JTP-3-Hydra-Features

## 1. NaFlex Architecture (Native Flexible Resolution)

Unlike previous models that resized every image to a fixed square (e.g., 384x384), JTP-3 uses **NaFlex** (Native Flexible) vision transformers.
*   **What it does:** It processes images at variable resolutions and aspect ratios by breaking them into a flexible number of patches (tokens).
*   **Benefit:** This preserves details in tall or wide images that would otherwise be squashed or cropped, leading to significantly higher tagging accuracy for non-square content.
*   **Control:** The `seqlen` widget allows you to control the maximum "sequence length" (number of patches), trading off between speed (lower seqlen) and fine detail resolution (higher seqlen).

## 2. The "Hydra" Head

The name "Hydra" comes from the model's custom multi-headed attention pooling mechanism.
*   **What it does:** Instead of a single classification head, it uses a complex routing system (`HydraPool`) with learned queries for different tag groups.
*   **Benefit:** This allows the model to efficiently handle a massive vocabulary of **7,504 tags** (including e621 tags and rating meta-tags) with much better separation between similar concepts compared to standard models.

## 3. Implications Engine

JTP-3 comes with a rich metadata system (`jtp-3-hydra-tags.csv`) that defines relationships between tags. The new node implements three logic modes:

*   **Inherit (Default):** If a specific tag (e.g., "blue_sky") is detected, the model automatically boosts the score of implied parent tags (e.g., "sky", "outdoor") to match it. This ensures logical consistency.
*   **Constrain:** Conversely, this prevents a specific tag from having a higher score than its required parent tag.
*   **Remove:** Automatically cleans up the output by removing redundant implied tags (e.g., outputting only "blue_sky" and hiding "sky" if requested).

## 4. Category Filtering

Because the model understands tag categories (Artist, Character, Species, Copyright, etc.), you can now filter outputs intelligently.
*   **Feature:** The `exclude_categories` input allows you to completely block entire classes of tags. For example, adding `artist, copyright` will ensure the output contains only descriptive visual tags, removing all meta-data tags.

## 5. Class Activation Maps (CAM)

JTP-3 natively supports visualizing "where" it sees a tag.
*   **Feature:** The node includes an `attention_map` output. This uses the model's attention layers to generate a heat map overlaying the image, showing exactly which pixels contributed most to the top detected tags.
*   **Note:** This is computationally expensive (requires a backward pass through the model), so it is generated only when specifically requested via the `cam_depth` parameter (higher depth = deeper visualization layers).

## 6. Calibrated Thresholding

*   **Score Range:** JTP-3 internally uses a calibrated score range from **-1.0 to 1.0**.
    *   `0.0` is the neutral "uncertain" point.
    *   Positive values (e.g., `0.5`) are confident detections.
    *   Negative values are confident rejections.
*   The node supports this full range, allowing for much finer control over sensitivity than the standard 0.0-1.0 probability range.