# Semantic Segmentation

## Models
1. U Net
2. FRR Net
3. Modified FRR Net - Added more skit connections and used a regular convolution layer from downscaling and ConvTranspose2d for upscaling.

## Results  

Image                                                    |  Probability Maps                      | Segmentations 
:-------------------------------------------------------:|:--------------------------------------:|:-------------------------:
<img src=Results/example1.png width="400" height="176"/> |  <img src=Results/example1_heat.png /> |  <img src=Results/example1_masks.png />
<img src=Results/example3.png width="400" height="176"/> |  <img src=Results/example3_heat.png /> |  <img src=Results/example3_masks.png />
<img src=Results/example4.png width="400" height="176"/> |  <img src=Results/example4_heat.png /> |  <img src=Results/example4_masks.png />

<img src=content_images/content_image5.jpg width="210" height="250"/> |  <img src=style_images/style5.jpg width="210" height="250"/> |  <img src=Result/target5.jpg width="210" height="250"/>
<img src=content_images/content_image3.jpg width="210" height="250"/> |  <img src=style_images/style3.jpg width="210" height="250"/> |  <img src=Result/target3.jpeg width="210" height="250"/>
<img src=content_images/content_image4.jpg width="210" height="250"/> |  <img src=style_images/style4.jpg width="210" height="250"/> |  <img src=Result/target4.jpg width="210" height="250"/>
<img src=content_images/content_image.jpg width="210" height="250"/>  |  <img src=style_images/style.jpg width="210" height="250"/>  |  <img src=Result/target.jpg width="210" height="250"/>
