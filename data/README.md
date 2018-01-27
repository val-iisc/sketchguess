
## Stroke sequence dataset
In our approach, we use sketches from [TU-Berlin dataset](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/). Each sketch is represented as a sequence of strokes. In the WordGuess-160 dataset, the stroke sequences are paired with corresponding guesswords. As a preprocessing step, each stroke sequence image is morphologically dilated ('thickened'). The dataset of thickened stroke sequence can be accessed at the following [link](https://drive.google.com/open?id=1unRU-zqqdfG3oT2a9Sx53AjlVaIqlBBn) as a .tar.gz archive. The archive contains two directories `sketches_png_css_thickened` and `sketches_png_css_thickened-sym`. The files should be accessed from the latter directory (i.e. `sketches_png_css_thickened-sym`).

## Auxiliary data for Sketchguess recurrent model

 *To test the Sketchguess recurrent model, 
 
    *Download [this](https://drive.google.com/open?id=1u9d432KdSRG22Zcsbhcahyah5UBhtZHr) compressed file and unzip it to     
       sketchguess/data/.
       *File info: w2v_data.zip
       *Contents : all_w2v.mat -- word embeddings stored per row
                  all_nouns.txt -- list of nouns corresponding to rows in all_w2v.mat
    
    *Download [this](https://drive.google.com/file/d/1KDnp4XV0YhxTZoa2QdmXVm-7GiD-3Hfv/view?usp=sharing) python pickled file that  
       contains CNN features that are input to the Sketchguess recurrent model. This file should be downloaded to sketchguess/data/
