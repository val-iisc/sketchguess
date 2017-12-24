
## Stroke sequence dataset
In our approach, we use sketches from TU-Berlin dataset. Each sketch is represented as a sequence of strokes. In the WordGuess-160 dataset, the stroke sequences are paired with corresponding guesswords. As a preprocessing step, each stroke sequence image is morphologically dilated (`thickened'). The dataset of thickened stroke sequence can be accessed at the following [link](https://drive.google.com/file/d/1Xc-PqRqzodrW5odpE1QQ4ZWXgspMv8lK/view?usp=sharing) as a .7z archive. 

### Instructions
1. The dataset has been archived as a 7-zip file. To extract files from the archive, you can use a command line tool such as [dtrx](https://askubuntu.com/a/586995)
2. The archive contains two directories `sketches_png_css_thickened` and `sketches_png_css_thickened-sym`
