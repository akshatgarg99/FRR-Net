# Semantic Segmentation

## Models
1. U Net
2. FRR Net
3. Modified FRR Net - Added more skit connections and used a regular convolution layer from downscaling and ConvTranspose2d for upscaling.

## Results  

Image                                                    |  Probability Maps                      | Segmentations 
:-------------------------------------------------------:|:--------------------------------------:|:-------------------------:
<img src=Results/example1.png width="370" height="176"/> |  <img src=Results/example1_heat.png /> |  <img src=Results/example1_masks.png />
<img src=Results/example3.png width="340" height="176"/> |  <img src=Results/example3_heat.png /> |  <img src=Results/example3_masks.png />
<img src=Results/example4.png width="370" height="176"/> |  <img src=Results/example4_heat.png /> |  <img src=Results/example4_masks.png />

<img src=Results/0001TP_009420.png width="370" height="176"/>      |  <img src=Results/bike1.png /> |  <img src=Results/boards1.png />
<img src=Results/buildings1.png >                                  |  <img src=Results/cars1.png /> |  <img src=Results/pavement1.png />
<img src=Results/poles1.png />                                     |  <img src=Results/road1.png /> |  <img src=Results/sky1.png />
